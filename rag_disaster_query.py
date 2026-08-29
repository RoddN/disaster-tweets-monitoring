"""
rag_disaster_query.py

Componente RAG (Retrieval-Augmented Generation) per i disaster tweets.

Utilizza ChromaDB come Vector Store per indicizzare i tweet tramite
embeddings vettoriali (SentenceTransformer) e permette di interrogare
il database in linguaggio naturale. La fase di generazione è delegata
a un LLM accessibile via OpenRouter.

Uso:
    export OPENROUTER_API_KEY="sk-or-v1-..."
    python rag_disaster_query.py
"""

import csv
import os

import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient

# Configurazione
CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "train.csv")
MONGO_URI = "mongodb://localhost:27017"
MONGO_DB = "bigdata_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
TOP_K = 5

# OpenRouter (API compatibile con OpenAI)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = "meta-llama/llama-3.1-8b-instruct:free"


def load_disaster_tweets():
    """
    Carica i tweet classificati come disastri reali.
    Prova da MongoDB (Speed Layer), altrimenti legge dal CSV.
    """
    # Prova MongoDB
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        client.admin.command("ping")
        db = client[MONGO_DB]

        if "disaster_tweets_raw" in db.list_collection_names():
            cursor = db["disaster_tweets_raw"].find(
                {"prediction": 1.0},
                {"_id": 0, "id": 1, "keyword": 1, "text": 1},
            )
            tweets = list(cursor)
            if tweets:
                print(f"[RAG] {len(tweets)} tweet caricati da MongoDB")
                return tweets
    except Exception:
        pass

    # Fallback: CSV
    if not os.path.exists(CSV_PATH):
        print(f"[RAG] File non trovato: {CSV_PATH}")
        return []

    tweets = []
    with open(CSV_PATH, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("target") == "1":
                tweets.append({
                    "id": row["id"],
                    "keyword": row.get("keyword", ""),
                    "text": row["text"],
                })

    print(f"[RAG] {len(tweets)} tweet caricati dal CSV")
    return tweets


def build_vector_store(tweets):
    """
    Indicizza i tweet in ChromaDB con embeddings vettoriali.
    ChromaDB usa internamente HNSW per la ricerca ANN (Approximate Nearest Neighbor).
    """
    print(f"[RAG] Modello di embedding: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    existing = [c.name for c in client.list_collections()]
    if "disaster_tweets" in existing:
        client.delete_collection("disaster_tweets")

    collection = client.create_collection(
        name="disaster_tweets",
        metadata={"hnsw:space": "cosine"},
    )

    texts = [t["text"] for t in tweets]
    ids = [str(t["id"]) for t in tweets]
    metadatas = [{"keyword": t.get("keyword", ""), "id": t["id"]} for t in tweets]

    BATCH_SIZE = 500
    for i in range(0, len(texts), BATCH_SIZE):
        end = min(i + BATCH_SIZE, len(texts))
        embeddings = model.encode(texts[i:end]).tolist()
        collection.add(
            embeddings=embeddings,
            documents=texts[i:end],
            metadatas=metadatas[i:end],
            ids=ids[i:end],
        )
        print(f"  Indicizzati {end}/{len(texts)} tweet")

    print(f"[RAG] Vector store pronto ({len(texts)} documenti)\n")
    return collection, model


def rag_query(collection, model, query, top_k=TOP_K):
    """
    Esegue una query RAG:
      1. Retrieval  — similarity search nel Vector Store
      2. Augmentation — formattazione del contesto
      3. Generation — risposta dell'LLM basata sul contesto
    """
    # Retrieval
    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    # Augmentation
    retrieved_docs = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    context_parts = []
    for i, (doc, dist, meta) in enumerate(zip(retrieved_docs, distances, metadatas), 1):
        similarity = 1 - dist
        keyword = meta.get("keyword", "N/A")
        context_parts.append(
            f"  [{i}] (similarita': {similarity:.2%}, keyword: {keyword})\n"
            f'      "{doc}"'
        )

    context = "\n".join(context_parts)

    # Generation (via OpenRouter)
    llm_answer = None
    if OPENROUTER_API_KEY:
        try:
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
            )

            prompt = (
                "Sei un assistente specializzato nell'analisi di disastri naturali. "
                "Basandoti ESCLUSIVAMENTE sui seguenti tweet reali recuperati dal "
                "database, rispondi alla domanda dell'utente in modo chiaro e conciso "
                "in italiano.\n\n"
                f"TWEET RECUPERATI:\n{context}\n\n"
                f"DOMANDA: {query}\n\n"
                "RISPOSTA:"
            )

            response = client.chat.completions.create(
                model=OPENROUTER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3,
            )
            llm_answer = response.choices[0].message.content.strip()
        except Exception as e:
            llm_answer = f"[Errore LLM: {e}]"

    return context, llm_answer


if __name__ == "__main__":
    print("=" * 60)
    print("  RAG — Disaster Tweets Query System")
    print(f"  Vector DB: ChromaDB | Embeddings: {EMBEDDING_MODEL}")
    if OPENROUTER_API_KEY:
        print(f"  LLM: {OPENROUTER_MODEL} (OpenRouter)")
    else:
        print("  LLM: non configurato (imposta OPENROUTER_API_KEY)")
    print("=" * 60)

    tweets = load_disaster_tweets()
    if not tweets:
        print("[RAG] Nessun tweet trovato.")
        exit(1)

    collection, model = build_vector_store(tweets)

    print("-" * 60)
    print("  Inserisci una domanda (oppure 'quit' per uscire)")
    print("  Esempi: earthquake damage, forest fire, flood rescue")
    print("-" * 60)

    while True:
        query = input("\nQuery > ").strip()
        if not query or query.lower() in ("quit", "exit", "q"):
            print("[RAG] Sessione terminata.")
            break

        context, llm_answer = rag_query(collection, model, query)

        print(f"\n--- Tweet recuperati (top {TOP_K}) ---\n")
        print(context)

        if llm_answer:
            print(f"\n--- Risposta LLM ---\n")
            print(f"  {llm_answer}")

        print()
