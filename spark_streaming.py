"""
spark_streaming.py

Speed Layer della Lambda Architecture.
Carica il modello addestrato dal Batch Layer e classifica i tweet in tempo reale
da Kafka tramite Spark Structured Streaming, salvando i risultati su MongoDB.
"""

import os
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from pyspark.ml import PipelineModel

# =============================================================================
# Configurazione
# =============================================================================
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "disaster_tweets"
MONGO_URI = "mongodb://localhost:27017"
MONGO_DATABASE = "bigdata_db"
MONGO_COLLECTION = "disaster_counts"

BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
MODEL_PATH = os.path.join(BASE_PATH, "models", "disaster_tweet_pipeline")
CHECKPOINT_PATH = os.path.join(BASE_PATH, "checkpoints", "disaster_stream")

# =============================================================================
# Spark Session
# =============================================================================
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

# =============================================================================
# Funzione di pulizia testo (identica al batch layer)
# =============================================================================
def clean_text_expr(col_expr):
    col = F.lower(F.coalesce(col_expr, F.lit("")))
    col = F.regexp_replace(col, r"http\S+|https\S+|www\.\S+", " ")
    col = F.regexp_replace(col, r"@\w+", " ")
    col = F.regexp_replace(col, r"#(\w+)", "$1")
    col = F.regexp_replace(col, r"[^a-z0-9\s]", " ")
    col = F.regexp_replace(col, r"\s+", " ")
    return F.trim(col)

# =============================================================================
# Speed Layer — Structured Streaming
# =============================================================================
if not os.path.exists(MODEL_PATH):
    print(f"[Errore] Modello non trovato in {MODEL_PATH}")
    print("Devi prima eseguire 'python spark_train_pipeline.py' per addestrare il modello.")
    exit(1)

# 1. Carica il modello
streaming_model = PipelineModel.load(MODEL_PATH)
print(f"[Speed Layer] Modello caricato da: {MODEL_PATH}")

# 2. Leggi da Kafka
tweet_schema = StructType([
    StructField("id", IntegerType(), True),
    StructField("keyword", StringType(), True),
    StructField("text", StringType(), True),
])

raw_stream = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
    .option("subscribe", KAFKA_TOPIC)
    .option("startingOffsets", "latest")
    .load()
)

# 3. Parsing e Preprocessing
parsed_stream = (
    raw_stream
    .selectExpr("CAST(value AS STRING) as json_str")
    .select(F.from_json(F.col("json_str"), tweet_schema).alias("data"))
    .select("data.*")
)

preprocessed_stream = (
    parsed_stream
    .withColumn(
        "keyword_clean",
        F.regexp_replace(F.coalesce(F.col("keyword"), F.lit("")), "%20", " "),
    )
    .withColumn("clean_text", clean_text_expr(F.col("text")))
    .withColumn(
        "clean_full_text",
        F.concat_ws(" ", F.col("keyword_clean"), F.col("clean_text")),
    )
)

# 4. Predizione
predictions_stream = streaming_model.transform(preprocessed_stream)

# 5. Tumbling Window (5 min) e Watermark (10 min)
predictions_with_time = predictions_stream.withColumn(
    "event_time", F.current_timestamp()
)

# Applichiamo una custom prediction in modo da usare la soglia ottimizzata a 0.35
# per minimizzare i falsi negativi, come testato in test_threshold.py
from pyspark.sql.types import DoubleType
extract_prob = F.udf(lambda v: float(v[1]), DoubleType())

predictions_opt = predictions_with_time.withColumn(
    "custom_prediction",
    F.when(extract_prob(F.col("probability")) >= 0.35, 1.0).otherwise(0.0)
)

windowed_counts = (
    predictions_opt
    .withWatermark("event_time", "10 minutes")
    .groupBy(F.window(F.col("event_time"), "5 minutes"))
    .agg(
        F.count("*").alias("total_count"),
        F.sum(F.when(F.col("custom_prediction") == 1.0, 1).otherwise(0)).alias("disaster_count"),
        F.collect_list(F.when(F.col("custom_prediction") == 1.0, F.col("id"))).alias("tweet_ids"),
    )
    .select(
        F.col("window.start").alias("window_start"),
        F.col("window.end").alias("window_end"),
        F.col("total_count"),
        F.col("disaster_count"),
        F.col("tweet_ids"),
    )
)

# 6. Scrittura su MongoDB
def write_to_mongodb(batch_df, batch_id):
    if batch_df.count() == 0:
        return

    enriched_df = (
        batch_df
        .withColumn("batch_id", F.lit(batch_id))
        .withColumn("window_start", F.col("window_start").cast("string"))
        .withColumn("window_end", F.col("window_end").cast("string"))
        .withColumn("_id", F.col("window_start"))
    )

    (
        enriched_df.write
        .format("mongodb")
        .mode("append")
        .option("connection.uri", MONGO_URI)
        .option("database", MONGO_DATABASE)
        .option("collection", MONGO_COLLECTION)
        .option("operationType", "Update")
        .option("idFieldList", "_id")
        .save()
    )
    print(f"[Speed Layer] Batch {batch_id}: "
          f"{batch_df.count()} finestre scritte su {MONGO_DATABASE}.{MONGO_COLLECTION}")


query = (
    windowed_counts
    .writeStream
    .outputMode("update")
    .foreachBatch(write_to_mongodb)
    .trigger(processingTime="5 seconds")
    .option("checkpointLocation", CHECKPOINT_PATH)
    .start()
)

print("[Speed Layer] Pipeline attiva, in attesa di tweet da Kafka...")
print(f"  Topic: {KAFKA_TOPIC}")
print(f"  Sink : {MONGO_URI}/{MONGO_DATABASE}.{MONGO_COLLECTION}")
print(f"  Modello con threshold ottimizzato (0.35) per minimizzare i falsi negativi")
query.awaitTermination()
