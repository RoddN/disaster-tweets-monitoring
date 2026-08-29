"""
test_threshold.py — Script per trovare il threshold ottimale
che minimizza i Falsi Negativi (massimizza la Recall).

Addestra il modello una sola volta, poi valuta diverse soglie
sul validation set mostrando la matrice di confusione per ciascuna.
"""

import os
from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

spark = (
    SparkSession.builder
    .appName("ThresholdTest")
    .master("local[*]")
    .config("spark.driver.memory", "2g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

# --- Caricamento e preprocessing ---
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "train.csv")

def clean_text_expr(col_expr):
    col = F.lower(F.coalesce(col_expr, F.lit("")))
    col = F.regexp_replace(col, r"http\S+|https\S+|www\.\S+", " ")
    col = F.regexp_replace(col, r"@\w+", " ")
    col = F.regexp_replace(col, r"#(\w+)", "$1")
    col = F.regexp_replace(col, r"[^a-z0-9\s]", " ")
    col = F.regexp_replace(col, r"\s+", " ")
    return F.trim(col)

raw_df = spark.read.option("header", "true").option("inferSchema", "true").csv(data_path)

preprocessed_df = (
    raw_df
    .withColumn("keyword_clean", F.regexp_replace(F.coalesce(F.col("keyword"), F.lit("")), "%20", " "))
    .withColumn("clean_text", clean_text_expr(F.col("text")))
    .withColumn("clean_full_text", F.concat_ws(" ", F.col("keyword_clean"), F.col("clean_text")))
    .select("id", "clean_full_text", F.col("target").cast("double").alias("label"))
    .dropna(subset=["label"])
)

train_df, val_df = preprocessed_df.randomSplit([0.8, 0.2], seed=42)

# --- Pipeline SENZA threshold (lo applichiamo manualmente dopo) ---
pipeline = Pipeline(stages=[
    RegexTokenizer(inputCol="clean_full_text", outputCol="words", pattern="\\s+"),
    StopWordsRemover(inputCol="words", outputCol="filtered_words"),
    HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000),
    IDF(inputCol="raw_features", outputCol="features"),
    LogisticRegression(featuresCol="features", labelCol="label", maxIter=1000, regParam=0.2),
])

print("Addestramento in corso...")
model = pipeline.fit(train_df)

# Predizioni con probabilita' (senza soglia fissa)
val_predictions = model.transform(val_df)
val_predictions.cache()

# --- Test di diversi threshold ---
thresholds = [0.5, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20]

print("\n" + "=" * 80)
print(f"{'Threshold':>10} | {'Recall':>8} | {'Precision':>10} | {'F1':>8} | {'Accuracy':>10} | {'FN':>6} | {'FP':>6} | {'TP':>6} | {'TN':>6}")
print("=" * 80)

for t in thresholds:
    # probability è un DenseVector [prob_class_0, prob_class_1]
    # Estraiamo prob_class_1 con una UDF
    from pyspark.sql.types import DoubleType
    from pyspark.ml.linalg import VectorUDT
    extract_prob = F.udf(lambda v: float(v[1]), DoubleType())

    evaluated = val_predictions.withColumn(
        "prob_disaster", extract_prob(F.col("probability"))
    ).withColumn(
        "custom_prediction",
        F.when(F.col("prob_disaster") >= t, 1.0).otherwise(0.0)
    )

    # Matrice di confusione
    tp = evaluated.filter((F.col("label") == 1.0) & (F.col("custom_prediction") == 1.0)).count()
    fn = evaluated.filter((F.col("label") == 1.0) & (F.col("custom_prediction") == 0.0)).count()
    fp = evaluated.filter((F.col("label") == 0.0) & (F.col("custom_prediction") == 1.0)).count()
    tn = evaluated.filter((F.col("label") == 0.0) & (F.col("custom_prediction") == 0.0)).count()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    print(f"{t:>10.2f} | {recall:>8.4f} | {precision:>10.4f} | {f1:>8.4f} | {accuracy:>10.4f} | {fn:>6} | {fp:>6} | {tp:>6} | {tn:>6}")

print("=" * 80)
print("\nLegenda:")
print("  FN = Falsi Negativi (disastri persi!)")
print("  FP = Falsi Positivi (falsi allarmi)")
print("  TP = Veri Positivi  (disastri correttamente rilevati)")
print("  TN = Veri Negativi  (non-disastri correttamente ignorati)")
print("\nObiettivo: minimizzare FN mantenendo un FP accettabile.\n")

spark.stop()
