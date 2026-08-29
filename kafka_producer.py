"""
kafka_producer.py

Simulatore di streaming per lo Speed Layer: legge i tweet dal file CSV di test
e li pubblica uno alla volta sul topic Kafka, simulando l'arrivo in tempo reale
di dati da una sorgente esterna (es. Twitter API).
"""

import csv
import json
import os
import time

from kafka import KafkaProducer

# Configurazione
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "disaster_tweets"
CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "test.csv")
SEND_DELAY_SECONDS = 0.5


def create_producer():
    """Crea un KafkaProducer configurato per l'invio di messaggi JSON."""
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks=1,
        retries=3,
    )


def read_and_publish(producer):
    """Legge il CSV riga per riga e pubblica ogni tweet sul topic Kafka."""
    if not os.path.exists(CSV_PATH):
        print(f"[Producer] File non trovato: {CSV_PATH}")
        return

    with open(CSV_PATH, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        count = 0

        for row in reader:
            message = {
                "id": int(row["id"]),
                "keyword": row.get("keyword", ""),
                "text": row.get("text", ""),
            }

            producer.send(KAFKA_TOPIC, value=message)
            count += 1

            if count % 50 == 0:
                print(f"[Producer] Inviati {count} tweet al topic '{KAFKA_TOPIC}'")

            time.sleep(SEND_DELAY_SECONDS)

    producer.flush()
    print(f"\n[Producer] Completato: {count} tweet inviati.")


if __name__ == "__main__":
    print(f"[Producer] Connessione a Kafka ({KAFKA_BOOTSTRAP_SERVERS})...")
    print(f"[Producer] Topic: {KAFKA_TOPIC}")
    print(f"[Producer] File sorgente: {CSV_PATH}")
    print(f"[Producer] Delay tra messaggi: {SEND_DELAY_SECONDS}s\n")

    producer = create_producer()

    try:
        read_and_publish(producer)
    except KeyboardInterrupt:
        print("\n[Producer] Interrotto dall'utente.")
    finally:
        producer.close()
        print("[Producer] Connessione chiusa.")
