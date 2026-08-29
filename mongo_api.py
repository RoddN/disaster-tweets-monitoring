"""
mongo_api.py

API Flask per il Serving Layer: espone i risultati aggregati dallo Speed Layer
(conteggi per finestra temporale) come endpoint JSON per Grafana.
"""

from flask import Flask, jsonify
from pymongo import MongoClient

app = Flask(__name__)

client = MongoClient("mongodb://localhost:27017")
db = client["bigdata_db"]

# Indice per ottimizzare l'ordinamento cronologico
db["disaster_counts"].create_index([("window_start", 1)])


@app.route("/disaster_counts")
def disaster_counts():
    """Restituisce i conteggi aggregati per finestra temporale."""
    docs = list(
        db["disaster_counts"]
        .find(
            {},
            {"_id": 0, "window_start": 1, "window_end": 1,
             "total_count": 1, "disaster_count": 1},
        )
        .sort("window_start", 1)
    )
    return jsonify(docs)


if __name__ == "__main__":
    print("[API] Avvio su http://localhost:5050")
    print("[API] Endpoint: GET /disaster_counts")
    app.run(host="0.0.0.0", port=5050)
