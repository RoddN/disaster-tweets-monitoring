"""
spark_train_pipeline.py

Pipeline di classificazione binaria dei tweet (Disastro Reale vs Non Disastro)
con architettura Lambda: Batch Layer per l'addestramento e Speed Layer per
l'elaborazione in tempo reale tramite Spark Structured Streaming.

Dataset: Kaggle "Natural Language Processing with Disaster Tweets"
"""

import os
import re
import shutil

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator,
)

# =============================================================================
# Spark Session
# =============================================================================
spark = (
    SparkSession.builder
    .appName("DisasterTweetClassifier")
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
# Configurazione path
# =============================================================================
# In produzione si usa HDFS:  BASE_PATH = "hdfs://localhost:9000/user/bigdata"
# Per il test locale usiamo il filesystem.
BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(BASE_PATH, exist_ok=True)

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "train.csv")

# =============================================================================
# Caricamento dati
# =============================================================================
raw_df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(data_path)
)

print(f"Righe nel dataset: {raw_df.count()}")
raw_df.printSchema()

# =============================================================================
# Preprocessing con funzioni native Spark SQL (no UDF Python)
# =============================================================================
# Riproduce la funzione clean_text() del notebook esplorativo usando
# espressioni native PySpark, evitando la serializzazione JVM-Python delle UDF.

def clean_text_expr(col_expr):
    """Pulizia del testo con regexp_replace native di Spark."""
    col = F.lower(F.coalesce(col_expr, F.lit("")))
    col = F.regexp_replace(col, r"http\S+|https\S+|www\.\S+", " ")
    col = F.regexp_replace(col, r"@\w+", " ")
    col = F.regexp_replace(col, r"#(\w+)", "$1")
    col = F.regexp_replace(col, r"[^a-z0-9\s]", " ")
    col = F.regexp_replace(col, r"\s+", " ")
    return F.trim(col)


preprocessed_df = (
    raw_df
    .withColumn(
        "keyword_clean",
        F.regexp_replace(F.coalesce(F.col("keyword"), F.lit("")), "%20", " "),
    )
    .withColumn("clean_text", clean_text_expr(F.col("text")))
    .withColumn(
        "clean_full_text",
        F.concat_ws(" ", F.col("keyword_clean"), F.col("clean_text")),
    )
    .select("id", "clean_full_text", F.col("target").cast("double").alias("label"))
    .dropna(subset=["label"])
)

preprocessed_df.show(5, truncate=60)

# =============================================================================
# Split train / validation (80/20)
# =============================================================================
train_df, val_df = preprocessed_df.randomSplit([0.8, 0.2], seed=42)

print(f"Train set: {train_df.count()} righe")
print(f"Validation set: {val_df.count()} righe")

# =============================================================================
# Pipeline MLlib (Tokenizer -> StopWords -> TF-IDF -> Logistic Regression)
# =============================================================================
tokenizer = RegexTokenizer(
    inputCol="clean_full_text", outputCol="words", pattern="\\s+"
)

stopwords_remover = StopWordsRemover(
    inputCol="words", outputCol="filtered_words"
)

hashing_tf = HashingTF(
    inputCol="filtered_words", outputCol="raw_features", numFeatures=10000
)

idf = IDF(inputCol="raw_features", outputCol="features")

lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    predictionCol="prediction",
    maxIter=1000,
    regParam=0.2,   # equivalente a C=5 in sklearn
)

pipeline = Pipeline(stages=[tokenizer, stopwords_remover, hashing_tf, idf, lr])

# =============================================================================
# Addestramento
# =============================================================================
print("\nAddestramento della pipeline...")
pipeline_model = pipeline.fit(train_df)
print("Pipeline addestrata.\n")

# =============================================================================
# Valutazione sul validation set
# =============================================================================
predictions = pipeline_model.transform(val_df)
predictions.select("id", "clean_full_text", "label", "prediction", "probability").show(10, truncate=50)

accuracy = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
).evaluate(predictions)

f1 = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="f1"
).evaluate(predictions)

precision = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="weightedPrecision"
).evaluate(predictions)

recall = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="weightedRecall"
).evaluate(predictions)

auc = BinaryClassificationEvaluator(
    labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
).evaluate(predictions)

print("Risultati sul validation set:")
print(f"  Accuracy  : {accuracy:.4f}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1-Score  : {f1:.4f}")
print(f"  AUC       : {auc:.4f}")

# =============================================================================
# Salvataggio del modello
# =============================================================================
model_path = os.path.join(BASE_PATH, "models", "disaster_tweet_pipeline")

if os.path.exists(model_path):
    shutil.rmtree(model_path)

pipeline_model.save(model_path)
print(f"\nModello salvato in: {model_path}")

loaded_model = PipelineModel.load(model_path)
print("Modello ricaricato correttamente.\n")

# =============================================================================
# Batch Views in formato Parquet
# =============================================================================
# Il Batch Layer della Lambda Architecture produce viste pre-computate
# (batch views) che vengono salvate in Parquet, un formato colonnare
# ottimizzato per query analitiche, con compressione Snappy e schema embedded.
# Le views sono partizionate per classe predetta per abilitare il partition pruning.

batch_views_path = os.path.join(BASE_PATH, "batch_views", "disaster_predictions")

all_predictions = pipeline_model.transform(preprocessed_df)

batch_views = all_predictions.select(
    "id", "clean_full_text", "label", "prediction", "probability"
)

batch_views.write.mode("overwrite").partitionBy("prediction").parquet(batch_views_path)

print(f"Batch views salvate in: {batch_views_path}")
print(f"  Formato: Parquet (Snappy)")
print(f"  Partizioni: prediction=0.0, prediction=1.0")
print(f"  Righe: {batch_views.count()}")

# =============================================================================
# Speed Layer — Structured Streaming da Kafka
# =============================================================================
# Legge i tweet in tempo reale dal topic Kafka, applica lo stesso modello ML
# addestrato nel batch, aggrega i risultati in finestre temporali (tumbling
# windows) e scrive i conteggi su MongoDB per la visualizzazione su Grafana.

STREAMING_DEMO = True

if STREAMING_DEMO:

    from pyspark.sql.types import StructType, StructField, IntegerType

    # Configurazione
    KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
    KAFKA_TOPIC = "disaster_tweets"
    MONGO_URI = "mongodb://localhost:27017"
    MONGO_DATABASE = "bigdata_db"
    MONGO_COLLECTION = "disaster_counts"

    # Schema JSON dei messaggi Kafka
    tweet_schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("keyword", StringType(), True),
        StructField("text", StringType(), True),
    ])

    # Caricamento modello addestrato
    streaming_model = PipelineModel.load(model_path)
    print("[Speed Layer] Modello caricato da:", model_path)

    # Lettura stream da Kafka
    raw_stream = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "latest")
        .load()
    )

    # Parsing JSON
    parsed_stream = (
        raw_stream
        .selectExpr("CAST(value AS STRING) as json_str")
        .select(F.from_json(F.col("json_str"), tweet_schema).alias("data"))
        .select("data.*")
    )

    # Preprocessing (identico al batch)
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

    # Predizione ML
    predictions_stream = streaming_model.transform(preprocessed_stream)

    # Tumbling Window con Watermark
    # Aggiungiamo il processing time come event_time perché il producer
    # non include un timestamp nei messaggi.
    predictions_with_time = predictions_stream.withColumn(
        "event_time", F.current_timestamp()
    )

    # Aggregazione: contiamo sia il totale dei tweet che quelli classificati
    # come disastro, raggruppati per finestre da 5 minuti.
    windowed_counts = (
        predictions_with_time
        .withWatermark("event_time", "10 minutes")
        .groupBy(F.window(F.col("event_time"), "5 minutes"))
        .agg(
            F.count("*").alias("total_count"),
            F.sum(F.when(F.col("prediction") == 1.0, 1).otherwise(0)).alias("disaster_count"),
            F.collect_list(F.when(F.col("prediction") == 1.0, F.col("id"))).alias("tweet_ids"),
        )
        .select(
            F.col("window.start").alias("window_start"),
            F.col("window.end").alias("window_end"),
            F.col("total_count"),
            F.col("disaster_count"),
            F.col("tweet_ids"),
        )
    )

    # Sink: scrittura su MongoDB con foreachBatch e upsert
    def write_to_mongodb(batch_df, batch_id):
        """Scrive ogni micro-batch su MongoDB con logica di upsert."""
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

    # Avvio della query di streaming
    query = (
        windowed_counts
        .writeStream
        .outputMode("update")
        .foreachBatch(write_to_mongodb)
        .trigger(processingTime="5 seconds")
        .option("checkpointLocation",
                os.path.join(BASE_PATH, "checkpoints", "disaster_stream"))
        .start()
    )

    print("[Speed Layer] Pipeline attiva, in attesa di tweet da Kafka...")
    print(f"  Topic: {KAFKA_TOPIC}")
    print(f"  Sink: {MONGO_URI}/{MONGO_DATABASE}.{MONGO_COLLECTION}")
    print(f"  Window: 5 min (tumbling) | Watermark: 10 min")
    query.awaitTermination()

# =============================================================================
spark.stop()
print("SparkSession chiusa.")
