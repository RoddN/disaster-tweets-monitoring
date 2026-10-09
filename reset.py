# reset.py
# Pulisce le collection MongoDB dello stream layer (view derivate).
# In Kappa le view sono ricostruibili dal log: questo script prepara per un replay.
# serve perche' le finestre si basano su current_timestamp(), quindi ogni
# run accumula dati nuovi invece di sovrascrivere quelli vecchi

import os

from pymongo import MongoClient
from config import MONGO_URI, MONGO_DB, COLL_COUNTS, COLL_RAW

if __name__ == "__main__":
    print("=" * 50)
    print("  Reset Stream Layer")
    print("=" * 50)

    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        client.admin.command("ping")
    except Exception as e:
        print(f"\n[Errore] Mongo non raggiungibile: {e}")
        exit(1)

    db = client[MONGO_DB]

    for name in [COLL_COUNTS, COLL_RAW]:
        coll = db[name]
        cnt = coll.count_documents({})
        res = coll.delete_many({})
        print(f"\n  {name}: {res.deleted_count}/{cnt} documenti rimossi")

    print("\n  Fatto. Puoi riavviare producer + streaming.")
    print("=" * 50)
    client.close()
