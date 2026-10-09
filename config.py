# config.py
# costanti condivise da tutti i layer, cosi' le cambio in un posto solo

import os

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# mongo
# Impostate MONGO_URI nel file .env (vedi .env.example)
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB = "disasterdb"
COLL_COUNTS = "disaster_counts"
COLL_RAW = "disaster_tweets_raw"      # Dove Spark salva i documenti elaborati

# OpenRouter (LLM)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# kafka
KAFKA_SERVER = "localhost:9092"
KAFKA_TOPIC = "disaster_tweets"

# path modello
MODEL_PATH = os.path.join(OUTPUT_DIR, "models", "disaster_tweet_pipeline")
MASTER_DATASET_PATH = os.path.join(OUTPUT_DIR, "master_dataset")

# dati
TRAIN_CSV = os.path.join(BASE_DIR, "data", "train.csv")
TEST_CSV = os.path.join(BASE_DIR, "data", "test.csv")
