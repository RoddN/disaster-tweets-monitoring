import os
from pymongo import MongoClient
import openai
from sentence_transformers import SentenceTransformer

from config import MONGO_URI, MONGO_DB, COLL_RAW, OPENROUTER_API_KEY

EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_NAME = "vector_index" 

print(f"[Sistema] Caricamento modello di embedding locale ({EMBED_MODEL})...")
embed_model = SentenceTransformer(EMBED_MODEL)

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]
coll = db[COLL_RAW]

llm_client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={"HTTP-Referer": "http://localhost:5050", "X-Title": "Disaster RAG"},
)

def query_atlas_vector_search(query, top_k=5):
    query_vector = embed_model.encode(query).tolist()
    pipeline = [
        {
            "$vectorSearch": {
                "index": INDEX_NAME,
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": 100,
                "limit": top_k,
                "filter": {"prediction": {"$eq": 1.0}}
            }
        },
        {
            "$project": {
                "_id": 0,
                "text": 1,
                "keyword": 1,
                "event_time": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]
    results = list(coll.aggregate(pipeline))
    
    parts = []
    for i, doc in enumerate(results, 1):
        score = doc.get("score", 0)
        kw = doc.get("keyword", "N/A")
        time_str = doc.get("event_time", "not available")
        text = doc.get("text", "")
        parts.append(f"  [{i}] (sim: {score:.2f}, time: {time_str}, kw: {kw})\n      \"{text}\"")
        
    return "\n".join(parts) if parts else ""

if __name__ == "__main__":
    print("=" * 60)
    print("  Real-Time RAG — Disaster Tweets (MongoDB Atlas Vector Search)")
    print("=" * 60)
    
    while True:
        q = input("\nQuery > ").strip()
        if not q or q.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break

        print("[1/2] Ricerca semantica real-time su MongoDB Atlas...")
        context = query_atlas_vector_search(q)
        
        if not context:
            print("Nessun tweet trovato su Atlas. Assicurati che lo streaming stia girando.")
            continue

        print("[2/2] Generazione risposta LLM tramite OpenRouter...\n")
        
        system_prompt = (
            "Sei un assistente per la gestione delle emergenze. "
            "Rispondi alla domanda basandoti ESCLUSIVAMENTE sui tweet forniti. "
            "Rispondi con massimo 3 frasi brevi. "
            "Se le fonti non bastano, scrivi: 'Informazioni insufficienti'."
        )
        
        prompt = f"[DOMANDA]\n{q}\n\n[CONTESTO]\n{context}"
        
        try:
            response = llm_client.chat.completions.create(
                model="meta-llama/llama-3.3-70b-instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=180,
                temperature=0.2,
            )
            print("🤖 ASSISTENTE:")
            print("=" * 60)
            print(response.choices[0].message.content)
            print("=" * 60)
            print("\n(Fonti live recuperate da Atlas:)")
            print(context)
            
        except Exception as e:
            print(f"❌ Errore LLM: {e}")