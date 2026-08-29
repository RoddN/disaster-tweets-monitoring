# Disaster Tweets - Big Data Lambda Architecture

This repository contains a full **Lambda Architecture** implementation designed to analyze and classify real-time streams of disaster-related tweets. The project identifies genuine natural disasters versus false alarms or metaphorical language, utilizing distributed computing technologies.

## Architecture

The system is built upon a standard Lambda Architecture, segmented into three main layers, plus an advanced semantic search module:

1. **Batch Layer (Apache Spark MLlib)**
   - Trains a `LogisticRegression` pipeline on historical dataset.
   - Saves the trained `PipelineModel` and produces pre-computed **Batch Views** in Apache Parquet format.
2. **Speed Layer (Apache Kafka & Spark Structured Streaming)**
   - Consumes real-time simulated tweets from a Kafka topic (`disaster_tweets`).
   - Uses Spark Structured Streaming to classify tweets in real-time.
   - Aggregates metrics using 5-minute Tumbling Windows and Watermarking to handle late data.
3. **Serving Layer (MongoDB & Flask API)**
   - Spark continuously upserts windowed counts to a MongoDB database.
   - A lightweight Flask API exposes the aggregated metrics as JSON endpoints.
4. **Visualization Layer (Grafana)**
   - Connects to the Flask API using a JSON Datasource.
   - Visualizes real-time metrics and disaster counts.
5. **Semantic RAG Module (ChromaDB + OpenRouter)**
   - Embeds real disaster tweets into a Vector Database (ChromaDB).
   - Uses Retrieval-Augmented Generation (RAG) with LLaMA 3.1 to answer user questions using *only* verified tweets as context.

## Technology Stack
* **Data Processing & ML:** Apache Spark (PySpark), MLlib, Spark Structured Streaming
* **Message Broker:** Apache Kafka, Zookeeper
* **Database:** MongoDB
* **API / Backend:** Python (Flask)
* **Visualization:** Grafana
* **AI / RAG:** ChromaDB, Sentence-Transformers, OpenRouter (LLaMA)
* **Orchestration:** Docker, Docker Compose

## How to Run

1. **Start the Infrastructure**
   ```bash
   docker compose up -d
   ```
   This will start Kafka, Zookeeper, MongoDB, and Grafana. Wait a few seconds for Kafka to initialize the topic.

2. **Run the Spark Pipeline (Batch + Speed)**
   Open a terminal, activate your virtual environment, and run:
   ```bash
   python spark_train_pipeline.py
   ```
   This will train the model, save it, and start listening for real-time Kafka streams.

3. **Start the Flask API**
   Open a new terminal and run:
   ```bash
   python mongo_api.py
   ```

4. **Simulate Real-time Tweet Ingestion**
   Open another terminal and run the producer:
   ```bash
   python kafka_producer.py
   ```

5. **View the Dashboard**
   Navigate to `http://localhost:3000` (User: `admin`, Pass: `admin`).

## RAG Semantic Query System
To test the semantic search module:
```bash
export OPENROUTER_API_KEY="your-openrouter-api-key"
python rag_disaster_query.py
```

## Dataset
Based on the Kaggle Dataset: [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started).
