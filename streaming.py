import os
import pandas as pd

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, ArrayType, FloatType
from pyspark.sql.functions import pandas_udf
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from sentence_transformers import SentenceTransformer

from text_preprocessing import clean_text_expr
from config import (
    KAFKA_SERVER, KAFKA_TOPIC, MONGO_URI, MONGO_DB, 
    COLL_COUNTS, COLL_RAW, MODEL_PATH, OUTPUT_DIR
)

STARTING_OFFSETS = os.getenv("STARTING_OFFSETS", "latest")
RUN_ID = os.getenv("RUN_ID", "live")

CKPT_COUNTS = os.path.join(OUTPUT_DIR, "checkpoints", RUN_ID, "disaster_stream")
CKPT_RAW = os.path.join(OUTPUT_DIR, "checkpoints", RUN_ID, "disaster_raw_stream")
# In Kappa, il replay usa lo stesso codice ma partendo da offset diversi.
# Checkpoint separati impediscono che il replay consumi gli offset della sessione live.

spark = (
    SparkSession.builder
    .appName("DisasterTweetStreaming")
    .master("local[*]")
    .config("spark.driver.memory", "2g")
    .config(
        "spark.jars.packages",
        "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,"
        "org.mongodb.spark:mongo-spark-connector_2.12:10.4.0",
    )
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

if not os.path.exists(MODEL_PATH):
    print(f"[Errore] Modello non trovato in {MODEL_PATH}")
    exit(1)

model = PipelineModel.load(MODEL_PATH)
lr = model.stages[-1]
THRESHOLD = lr.getThreshold()

@pandas_udf(ArrayType(FloatType()))
def compute_embeddings(text_series: pd.Series) -> pd.Series:
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = emb_model.encode(text_series.tolist()).tolist()
    return pd.Series(embeddings)

tweet_schema = StructType([
    StructField("id", IntegerType(), True),
    StructField("keyword", StringType(), True),
    StructField("text", StringType(), True),
])

raw_stream = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_SERVER)
    .option("subscribe", KAFKA_TOPIC)
    .option("startingOffsets", STARTING_OFFSETS)
    .load()
)

parsed = (
    raw_stream
    .selectExpr("CAST(value AS STRING) as json_str", "timestamp as kafka_timestamp")
    .select(F.from_json(F.col("json_str"), tweet_schema).alias("data"), "kafka_timestamp")
    .select("data.*", "kafka_timestamp")
)

preprocessed = (
    parsed
    .withColumn("keyword_clean", F.regexp_replace(F.coalesce(F.col("keyword"), F.lit("")), "%20", " "))
    .withColumn("clean_text", clean_text_expr(F.col("text")))
    .withColumn("clean_full_text", F.concat_ws(" ", F.col("keyword_clean"), F.col("clean_text")))
)

preds = model.transform(preprocessed)
preds = preds.withColumn("event_time", F.coalesce(F.col("kafka_timestamp"), F.current_timestamp()))
preds = preds.withColumn("custom_prediction", F.when(vector_to_array(F.col("probability"))[1] >= THRESHOLD, 1.0).otherwise(0.0))

windowed = (
    preds
    .withWatermark("event_time", "2 minutes")
    .groupBy(F.window(F.col("event_time"), "1 minute"))
    .agg(
        F.count("*").alias("total_count"),
        F.sum(F.when(F.col("custom_prediction") == 1.0, 1).otherwise(0)).alias("disaster_count"),
        F.collect_list(F.when(F.col("custom_prediction") == 1.0, F.col("id"))).alias("tweet_ids"),
    )
    .select(
        F.col("window.start").alias("window_start"),
        F.col("window.end").alias("window_end"),
        "total_count", "disaster_count", "tweet_ids",
    )
)

def write_counts(batch_df, batch_id):
    if batch_df.count() == 0: return
    df = (
        batch_df
        .withColumn("batch_id", F.lit(batch_id))
        .withColumn("window_start", F.col("window_start").cast("string"))
        .withColumn("window_end", F.col("window_end").cast("string"))
        .withColumn("_id", F.col("window_start"))
    )
    (
        df.write.format("mongodb").mode("append")
        .option("connection.uri", MONGO_URI)
        .option("database", MONGO_DB)
        .option("collection", COLL_COUNTS)
        .option("operationType", "Update")
        .option("idFieldList", "_id")
        .save()
    )

def write_raw_tweets(batch_df, batch_id):
    if batch_df.count() == 0: return
    df = batch_df.select(
        F.col("id").cast("string").alias("_id"),
        F.col("id"), F.col("text"), F.col("keyword"),
        F.col("custom_prediction").alias("prediction"),
        vector_to_array(F.col("probability"))[1].alias("disaster_probability"),
        F.col("event_time"),
    ).withColumn(
        "embedding",
        F.when(F.col("prediction") == 1.0, compute_embeddings(F.col("text"))).otherwise(F.lit(None))
    )
    (
        df.write.format("mongodb").mode("append")
        .option("connection.uri", MONGO_URI)
        .option("database", MONGO_DB)
        .option("collection", COLL_RAW)
        .option("operationType", "Update")
        .option("idFieldList", "_id")
        .save()
    )
    print(f"[Stream] Batch {batch_id}: {df.count()} tweet inviati ad Atlas -> {COLL_RAW}")

q1 = windowed.writeStream.outputMode("update").foreachBatch(write_counts).trigger(processingTime="5 seconds").option("checkpointLocation", CKPT_COUNTS).start()
q2 = preds.writeStream.outputMode("append").foreachBatch(write_raw_tweets).trigger(processingTime="5 seconds").option("checkpointLocation", CKPT_RAW).start()

print("[Stream] Pipeline attiva (Atlas Vector Search)...")
spark.streams.awaitAnyTermination()