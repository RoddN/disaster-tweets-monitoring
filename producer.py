# speed/producer.py
# legge i tweet dal CSV di test e li manda su Kafka uno alla volta
# simula l'arrivo real-time (tipo Twitter API)

import csv
import json
import os
import time


from kafka import KafkaProducer
from config import KAFKA_SERVER, KAFKA_TOPIC, TEST_CSV

DELAY = 1.5  # secondi tra un messaggio e l'altro


def make_producer():
    return KafkaProducer(
        bootstrap_servers=KAFKA_SERVER,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks=1,
        retries=3,
    )


def publish_tweets(producer):
    if not os.path.exists(TEST_CSV):
        print(f"[Producer] File non trovato: {TEST_CSV}")
        return

    with open(TEST_CSV, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cnt = 0

        for row in reader:
            msg = {
                "id": int(row["id"]),
                "keyword": row.get("keyword", ""),
                "text": row.get("text", ""),
            }
            producer.send(KAFKA_TOPIC, value=msg)
            cnt += 1

            if cnt % 50 == 0:
                print(f"[Producer] Inviati {cnt} tweet")

            time.sleep(DELAY)

    producer.flush()
    print(f"\n[Producer] Finito: {cnt} tweet inviati.")


if __name__ == "__main__":
    print(f"[Producer] Kafka: {KAFKA_SERVER}")
    print(f"[Producer] Topic: {KAFKA_TOPIC}")
    print(f"[Producer] CSV: {TEST_CSV}")
    print(f"[Producer] Delay: {DELAY}s\n")

    producer = make_producer()

    try:
        publish_tweets(producer)
    except KeyboardInterrupt:
        print("\n[Producer] Interrotto.")
    finally:
        producer.close()
        print("[Producer] Chiuso.")
