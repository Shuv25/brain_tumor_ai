import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import requests

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("brain-tumor")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def process_text(user_query):
    embedded_query = embedding_model.encode(user_query).tolist()

    results = index.query(
        vector=embedded_query,
        top_k=3,
        include_metadata=True
    )


    context = "\n\n".join([match["metadata"]["text"] for match in results["matches"]])


    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    groq_payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful medical assistant. Use the following brain tumor information to answer the user."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}"}
        ],
        "model": "mistral-saba-24b" 
    }

    groq_response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=groq_payload)

    try:
        return groq_response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error processing response: {e}"
