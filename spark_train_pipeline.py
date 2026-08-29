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
spark.stop()
print("SparkSession chiusa.")
