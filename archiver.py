# archiver.py (optional, offline export)
# Esporta il log Kafka (via i tweet elaborati dallo stream) in Parquet per
# analisi storiche e dataset di annotazione.
# In Kappa non e' parte della pipeline: il log Kafka e' il master dataset.

import os
from datetime import datetime

from pymongo import MongoClient
from bson import ObjectId
import pyarrow as pa
import pyarrow.parquet as pq

from config import MONGO_URI, MONGO_DB, COLL_RAW, MASTER_DATASET_PATH


def fetch_unarchived_tweets(db):
    """Prende i tweet disastro non ancora finiti nel master dataset."""
    docs = list(db[COLL_RAW].find({"archived": {"$ne": True}}))
    for doc in docs:
        doc["mongo_id"] = str(doc.pop("_id"))
    return docs


def write_partition(docs):
    os.makedirs(MASTER_DATASET_PATH, exist_ok=True)
    table = pa.Table.from_pylist(docs)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(MASTER_DATASET_PATH, f"part_{ts}.parquet")
    pq.write_table(table, out_path)
    return out_path


def mark_archived(db, docs):
    ids = [ObjectId(doc["mongo_id"]) for doc in docs]
    db[COLL_RAW].update_many({"_id": {"$in": ids}}, {"$set": {"archived": True}})


if __name__ == "__main__":
    print("=" * 50)
    print("  Archiver — Stream Layer -> Master Dataset (offline export)")
    print("=" * 50)

    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        client.admin.command("ping")
    except Exception as e:
        print(f"\n[Archiver] Mongo non raggiungibile: {e}")
        exit(1)

    db = client[MONGO_DB]
    docs = fetch_unarchived_tweets(db)

    if not docs:
        print("\n[Archiver] Nessun tweet nuovo da archiviare.")
        client.close()
        exit(0)

    out_path = write_partition(docs)
    mark_archived(db, docs)

    print(f"\n[Archiver] {len(docs)} tweet archiviati -> {out_path}")
    client.close()
