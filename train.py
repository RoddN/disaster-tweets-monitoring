# train.py
# Training offline — addestra un classificatore binario sui disaster tweets
# e salva il modello. In Kappa, la ricomputazione avviene rilanciando lo stream.

import os
import shutil


from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator,
)

from text_preprocessing import clean_text_expr
from config import TRAIN_CSV, MODEL_PATH, OUTPUT_DIR

# plots.py è nella stessa cartella batch/
import plots

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

os.makedirs(OUTPUT_DIR, exist_ok=True)

raw_df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(TRAIN_CSV)
)

print(f"Righe nel dataset: {raw_df.count()}")
raw_df.printSchema()

df = (
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

df.show(5, truncate=60)

train_df, val_df = df.randomSplit([0.8, 0.2], seed=42)
print(f"Train: {train_df.count()} righe")
print(f"Val: {val_df.count()} righe")

# pipeline: tokenizer -> stopwords -> tf-idf -> logistic regression
tokenizer = RegexTokenizer(
    inputCol="clean_full_text", outputCol="words", pattern="\\s+"
)
stopwords = StopWordsRemover(inputCol="words", outputCol="filtered_words")

cv = CountVectorizer(
    inputCol="filtered_words", outputCol="raw_features", vocabSize=10000, minDF=5
)
idf = IDF(inputCol="raw_features", outputCol="features")

lr = LogisticRegression(
    featuresCol="features", labelCol="label", predictionCol="prediction",
    maxIter=1000,
    regParam=0.2,  # equivalente a C=5 in sklearn
)

pipeline = Pipeline(stages=[tokenizer, stopwords, cv, idf, lr])

print("\nAddestramento...")
pipeline_model = pipeline.fit(train_df)
print("Fatto.\n")

# con CountVectorizer posso vedere quali parole pesano di piu'
print("--- Coefficienti ---")
cv_model = pipeline_model.stages[2]
lr_model = pipeline_model.stages[-1]

vocab = cv_model.vocabulary
weights = lr_model.coefficients.toArray()

ww = sorted(zip(vocab, weights), key=lambda x: x[1], reverse=True)

print("Top 10 parole -> DISASTRO:")
for w, weight in ww[:10]:
    print(f"  {w:<15}: {weight:.4f}")

print("\nTop 10 parole -> NON DISASTRO:")
for w, weight in ww[-10:]:
    print(f"  {w:<15}: {weight:.4f}")
print("---\n")

# cerco la soglia che massimizza F1 sulla classe disastro (1.0)
# F1 bilancia equamente precision e recall
print("Threshold tuning (max F1 sulla classe 1)...")
best_t = 0.5
best_f1 = 0.0

f1_eval = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction",
    metricName="fMeasureByLabel", metricLabel=1.0, beta=1.0
)
prec_eval = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction",
    metricName="precisionByLabel", metricLabel=1.0
)
rec_eval = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction",
    metricName="recallByLabel", metricLabel=1.0
)

thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
f1_scores, prec_scores, rec_scores = [], [], []

for t in thresholds:
    lr_model.setThreshold(t)
    tmp = pipeline_model.transform(val_df)
    f1 = f1_eval.evaluate(tmp)
    p = prec_eval.evaluate(tmp)
    r = rec_eval.evaluate(tmp)
    f1_scores.append(f1)
    prec_scores.append(p)
    rec_scores.append(r)
    print(f"  t={t:.2f} -> F1: {f1:.4f}  Prec: {p:.4f}  Rec: {r:.4f}")
    if f1 > best_f1:
        best_f1 = f1
        best_t = t

print(f"\nSoglia migliore: {best_t:.2f} (F1: {best_f1:.4f})\n")

# salvo la soglia nel modello -> lo streaming la legge da qui al prossimo avvio
# (se lo streaming e' gia' attivo va riavviato)
lr_model.setThreshold(best_t)

# valutazione finale con soglia ottimizzata
preds = pipeline_model.transform(val_df)
preds.select("id", "clean_full_text", "label", "prediction", "probability").show(10, truncate=50)

accuracy = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
).evaluate(preds)

auc = BinaryClassificationEvaluator(
    labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
).evaluate(preds)

rec1 = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction",
    metricName="recallByLabel", metricLabel=1.0
).evaluate(preds)

prec1 = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction",
    metricName="precisionByLabel", metricLabel=1.0
).evaluate(preds)

print(f"Risultati (soglia {best_t:.2f}):")
print(f"  Accuracy : {accuracy:.4f}")
print(f"  AUC      : {auc:.4f}")
print(f"  --- Classe 1 (Disastro) ---")
print(f"  Recall   : {rec1:.4f}")
print(f"  Precision: {prec1:.4f}")
print(f"  F1       : {best_f1:.4f}  <- massimizzato")

from pyspark.mllib.evaluation import MulticlassMetrics

# Otteniamo la confusion matrix tramite l'oggetto MulticlassMetrics di Spark
# (esattamente come mostrato nei laboratori del corso per la Logistic Regression)
predictionAndLabels = preds.select("prediction", "label").rdd.map(lambda r: (float(r[0]), float(r[1])))
metrics = MulticlassMetrics(predictionAndLabels)
cm = metrics.confusionMatrix().toArray()

# in Spark MulticlassMetrics per classificazione binaria (0.0 e 1.0):
# cm[0][0] = TN, cm[0][1] = FP
# cm[1][0] = FN, cm[1][1] = TP
tn, fp = cm[0][0], cm[0][1]
fn, tp = cm[1][0], cm[1][1]

plots_dir = os.path.join(OUTPUT_DIR, "plots")
plots.generate(
    tp, fp, fn, tn, thresholds, prec_scores, rec_scores, f1_scores, best_t, best_f1,
    accuracy, auc, rec1, prec1, plots_dir
)

# salvo il modello
if os.path.exists(MODEL_PATH):
    shutil.rmtree(MODEL_PATH)

pipeline_model.save(MODEL_PATH)
print(f"\nModello salvato in: {MODEL_PATH}")

# verifica che si ricarichi
loaded = PipelineModel.load(MODEL_PATH)
print("Modello ricaricato ok.\n")

spark.stop()
print("\n[Train] Modello addestrato e salvato. Kappa: usa lo stream per ricomputare.")
