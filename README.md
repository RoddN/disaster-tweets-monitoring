# Disaster Tweets - Big Data Lambda Architecture

This project implements a Lambda Architecture for processing and classifying tweets related to natural disasters.

The system uses historical data to train a machine learning model and then applies it to tweets coming from a real-time Kafka stream. The results are stored in MongoDB and displayed through a Grafana dashboard. A separate RAG-based module allows users to search through real disaster tweets using natural language.

## Architecture

The project is divided into the following components:

### 1. Batch Layer

The batch layer uses Apache Spark and MLlib to train a Logistic Regression model on the historical disaster tweets dataset.

The trained model is saved as a Spark `PipelineModel`, while the results of the batch processing are stored in Parquet format.

### 2. Speed Layer

The speed layer handles tweets as they arrive through Apache Kafka.

Spark Structured Streaming reads tweets from the `disaster_tweets` topic and uses the trained model to classify them in real time.

The streaming job also calculates statistics using 5-minute tumbling windows. Watermarking is used to deal with tweets that arrive late.

### 3. Serving Layer

The streaming results are stored in MongoDB.

A small Flask API provides access to the aggregated data through JSON endpoints. This API is then used by Grafana to display the results.

### 4. Visualization

Grafana is used to create a dashboard showing the data produced by the streaming pipeline, including disaster-related tweet counts and other aggregated metrics.

### 5. RAG Module

The project also includes a semantic search module based on ChromaDB and OpenRouter.

Tweets classified as real disasters are converted into embeddings and stored in a vector database. When a user asks a question, the system retrieves the most relevant tweets and sends them as context to an LLaMA model through OpenRouter.

This allows users to ask questions about the collected disaster tweets using natural language.

## Technology Stack
- **Data processing and ML:** Apache Spark, PySpark, MLlib, Spark Structured Streaming
- **Message broker:** Apache Kafka, Zookeeper
- **Database:** MongoDB
- **API:** Python, Flask
- **Visualization:** Grafana
- **Semantic search:** ChromaDB, Sentence-Transformers, OpenRouter, LLaMA
- **Infrastructure:** Docker, Docker Compose

## How to Run

### 1. Start the infrastructure

Start the required services with:

```bash
docker compose up -d
```

This starts Kafka, Zookeeper, MongoDB and Grafana.

### 2. Start the Spark pipeline

Activate the Python virtual environment and run:

```bash
python spark_train_pipeline.py
```

The script trains the model, saves it and starts the Spark streaming job that listens for messages from Kafka.

### 3. Start the Flask API

In a separate terminal:

```bash
python mongo_api.py
```

### 4. Start the Kafka producer

To simulate the incoming tweets:

```bash
python kafka_producer.py
```

The producer sends tweets to the `disaster_tweets` Kafka topic.

### 5. Open Grafana

The dashboard is available at:

http://localhost:3000

**Default credentials:**
- Username: `admin`
- Password: `admin`

## RAG Semantic Search

To use the semantic search module, set your OpenRouter API key:

```bash
export OPENROUTER_API_KEY="your-openrouter-api-key"
```

Then run:

```bash
python rag_disaster_query.py
```

The script can be used to ask questions about the disaster tweets stored in the vector database.

## Dataset

The project uses the Natural Language Processing with Disaster Tweets dataset from Kaggle:

https://www.kaggle.com/c/nlp-getting-started
