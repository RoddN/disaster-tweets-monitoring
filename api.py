# api.py
# Espone i dati su endpoint JSON per Grafana.
# In Kappa esiste una sola vista: quella derivata dallo stream (mongo).

from flask import Flask, jsonify
from pymongo import MongoClient

from config import MONGO_URI, MONGO_DB, COLL_COUNTS

app = Flask(__name__)

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]
db[COLL_COUNTS].create_index([("window_start", 1)])


@app.route("/")
def index():
    return jsonify({
        "message": "Kappa Architecture API is running",
        "endpoints": ["/disaster_counts", "/disaster_summary"]
    })


@app.route("/disaster_counts")
def disaster_counts():
    """Conteggi per finestra temporale (stream layer)."""
    docs = list(
        db[COLL_COUNTS]
        .find({}, {"_id": 0, "window_start": 1, "window_end": 1,
                    "total_count": 1, "disaster_count": 1})
        .sort("window_start", 1)
    )
    return jsonify(docs)


@app.route("/disaster_summary")
def disaster_summary():
    """Vista unificata dello stream processato (Kappa)."""

    speed_docs = list(
        db[COLL_COUNTS]
        .find({}, {"_id": 0, "window_start": 1, "window_end": 1,
                    "total_count": 1, "disaster_count": 1})
        .sort("window_start", 1)
    )

    sp_total = sum(d.get("total_count", 0) for d in speed_docs)
    sp_disaster = sum(d.get("disaster_count", 0) for d in speed_docs)

    result = {
        "total_tweets": sp_total,
        "disaster_count": sp_disaster,
        "not_disaster_count": sp_total - sp_disaster,
        "windows": len(speed_docs),
        "source": "stream_view (MongoDB, Kappa)",
    }

    if sp_total > 0:
        result["disaster_ratio"] = round(sp_disaster / sp_total, 4)
    else:
        result["disaster_ratio"] = 0

    return jsonify(result)


if __name__ == "__main__":
    print("[API] http://localhost:5050")
    print("[API] GET /disaster_counts")
    print("[API] GET /disaster_summary")
    app.run(host="0.0.0.0", port=5050)
