# Disaster Tweets: Real-Time Classification with Kafka and Spark

This project classifies tweets as "about a real disaster" or "not", as they arrive. A Spark model is trained once on a labelled dataset, then a Spark Streaming job applies it to every tweet that comes in through Kafka. The results land in MongoDB, a small Flask API exposes them, and a Grafana dashboard shows what is happening. A separate search module lets you ask questions in natural language over the disaster tweets the system has found.

## How it works

```
                 train.csv (Kaggle)
                        |
                  train.py  (Spark MLlib, run once)
                        |
                  saved model
                        |
test.csv --> producer.py --> Kafka topic: disaster_tweets
                                  |
                       streaming.py (Spark Structured Streaming)
                          |                     |
              per-minute counts          raw tweets + predictions
                          \                     /
                           MongoDB (disasterdb)
                              |            |
                          api.py        rag.py
                        (Flask, :5050)  (semantic search)
                              |
                         Grafana (:3000)
```

The design follows the Kappa approach: there is one processing path, the Kafka stream, and every view is derived from it. The API reads from the same MongoDB collections that the stream writes to. There is no separate batch layer to merge with.

The one exception is model training. `train.py` runs as an offline job on the historical dataset and saves a Spark `PipelineModel`. The streaming job loads that model and uses it for every incoming tweet. To change the model, retrain it and restart the stream.

### Components

- **`train.py`**: reads the labelled training set, cleans the text, and trains a Logistic Regression classifier with a TF-IDF pipeline. It saves the model and prints evaluation plots, which `plots.py` generates.
- **`producer.py`**: replays the test tweets into Kafka one at a time, with a short delay between them, to simulate a live feed.
- **`streaming.py`**: consumes the Kafka topic, classifies each tweet, and writes two things to MongoDB every five seconds: per-minute counts of total and disaster tweets, and the raw tweets with their predictions and embeddings.
- **`api.py`**: a small Flask API that serves the aggregated counts to Grafana.
- **`rag.py`**: semantic search over the stored tweets using MongoDB Atlas Vector Search and an LLM served through OpenRouter.
- **`archiver.py`**: optional. Exports tweets that have not been archived yet to Parquet, for offline analysis or labelling.
- **`reset.py`**: clears the stream collections so you can start a clean run.
- **`config.py`**: shared settings such as the MongoDB URI, Kafka address, topic name and file paths.

## Requirements

- Python 3.10+ and Java (required by PySpark)
- Docker and Docker Compose
- The Kaggle dataset: [NLP with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started). Place `train.csv` and `test.csv` in the `data/` folder.
- Optional, for semantic search: a MongoDB Atlas cluster with a vector search index, and an OpenRouter API key

## Setup

Create a virtual environment and install the dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Start Kafka, Zookeeper, MongoDB and Grafana:

```bash
docker compose up -d
```

The compose file creates the `disaster_tweets` topic automatically.

## Running the pipeline

Run these steps in order.

**1. Train the model.** This writes the model to `output/models/`.

```bash
python train.py
```

**2. Start the streaming job.** Leave it running.

```bash
python streaming.py
```

**3. Start the producer.** In a second terminal, this feeds the test tweets into Kafka.

```bash
python producer.py
```

**4. Start the API.** In a third terminal.

```bash
python api.py
```

Useful endpoints:

- `GET /disaster_counts`: counts per one-minute window
- `GET /disaster_summary`: totals, disaster ratio and number of windows

**5. Open the dashboard.** Go to http://localhost:3000 and log in with `admin` / `admin`. The dashboard reads from the Flask API through the JSON datasource plugin.

To start over, stop the streaming job and producer, then run:

```bash
python reset.py
```

## Configuration

Most settings live in `config.py`. Two things you will probably need to change:

Secrets and machine-specific settings go in a `.env` file in the project root. The file is listed in `.gitignore`, so it stays out of version control. Create it from the template:

```bash
cp .env.example .env
```

Then edit `.env`:

- **`MONGO_URI`**: the MongoDB connection string. The template points to the MongoDB container from `docker-compose.yml` (`mongodb://localhost:27017/`). Use your Atlas URI here if you run against Atlas.
- **`OPENROUTER_API_KEY`**: needed only for `rag.py`.

Do not commit `.env`. If a real credential has been shared or pushed anywhere, rotate it.

## Semantic search (RAG)

`rag.py` embeds your question with a local Sentence-Transformers model (`all-MiniLM-L6-v2`), finds the most similar stored tweets with MongoDB Atlas Vector Search, and passes them to an LLM as context. This requires a vector search index named `vector_index` on the raw tweets collection.

```bash
python rag.py
```

Then ask questions in natural language, for example: "Are there requests for ambulances?"

## Project layout

```
api.py              Flask API for Grafana
archiver.py         Optional export of raw tweets to Parquet
config.py           Shared settings
docker-compose.yml  Kafka, Zookeeper, MongoDB, Grafana
grafana/            Dashboard provisioning
plots.py            Evaluation plots for training
producer.py         Replays test tweets into Kafka
rag.py              Semantic search over tweets
reset.py            Clears stream collections
streaming.py        Spark Structured Streaming job
text_preprocessing.py  Text cleaning used by training and streaming
train.py            Model training
```

## Dataset

The project uses the Natural Language Processing with Disaster Tweets dataset from Kaggle:
https://www.kaggle.com/c/nlp-getting-started
