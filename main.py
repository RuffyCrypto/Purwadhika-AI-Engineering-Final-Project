# =====================================================
# LOAD ENV (HARUS PALING ATAS)
# =====================================================
from dotenv import load_dotenv
load_dotenv()

# =====================================================
# IMPORTS
# =====================================================
import os
import sqlite3
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openai import OpenAI
from qdrant_client import QdrantClient

# =====================================================
# APP INIT
# =====================================================
app = FastAPI(title="Olist Multi-Agent AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# ENV VARS (JANGAN CRASH SAAT STARTUP)
# =====================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not OPENAI_API_KEY:
    print("⚠️ WARNING: OPENAI_API_KEY belum diset")

# =====================================================
# CLIENTS (LAZY & AMAN)
# =====================================================
openai_client = None
if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

qdrant = None
if QDRANT_URL and QDRANT_API_KEY:
    qdrant = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=30.0
    )

DB_PATH = "olist.db"  # optional

# =====================================================
# REQUEST MODEL
# =====================================================
class ChatRequest(BaseModel):
    query: str

# =====================================================
# UTILITIES
# =====================================================
def get_db():
    if not os.path.exists(DB_PATH):
        return None
    return sqlite3.connect(DB_PATH)

def embed_text(text: str):
    if not openai_client:
        return None

    return openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

# =====================================================
# SQL AGENT
# =====================================================
def sql_agent(query: str):
    q = query.lower()

    if not any(k in q for k in ["harga", "price", "seller", "kota", "lokasi"]):
        return None

    conn = get_db()
    if conn is None:
        return None

    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT product_id, price, seller_city
            FROM products
            LIMIT 5
        """)
        rows = cur.fetchall()
        conn.close()

        if not rows:
            return None

        answer = "📊 Data produk dari database:\n"
        for r in rows:
            answer += f"- Produk {r[0]} | Harga {r[1]} | Kota seller {r[2]}\n"

        return {
            "answer": answer,
            "source": "SQL-Agent"
        }

    except Exception as e:
        print("SQL Agent error:", e)
        return None

# =====================================================
# RAG AGENT
# =====================================================
def rag_agent(query: str):
    if not qdrant or not openai_client:
        return None

    try:
        vector = embed_text(query)
        if vector is None:
            return None

        hits = qdrant.search(
            collection_name="olist_products",
            query_vector=vector,
            limit=3
        )

        if not hits:
            return None

        context = "\n".join(
            [h.payload.get("text", "") for h in hits if h.payload]
        )

        prompt = f"""
Gunakan konteks berikut untuk menjawab pertanyaan pengguna.

Konteks:
{context}

Pertanyaan:
{query}
"""

        res = openai_client.chat.completions.create(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        return {
            "answer": res.choices[0].message.content,
            "source": "RAG-Agent"
        }

    except Exception as e:
        print("RAG Agent error:", e)
        return None

# =====================================================
# LLM FALLBACK AGENT (SELALU AMAN)
# =====================================================
def llm_fallback(query: str):
    if not openai_client:
        return {
            "answer": "LLM belum tersedia karena API key belum diset.",
            "source": "System"
        }

    res = openai_client.chat.completions.create(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": "Anda adalah asisten e-commerce Olist."},
            {"role": "user", "content": query}
        ],
        temperature=0.3
    )

    return {
        "answer": res.choices[0].message.content,
        "source": "LLM-Fallback"
    }

# =====================================================
# ROUTER AGENT
# =====================================================
def router_agent(query: str):
    sql_res = sql_agent(query)
    if sql_res:
        return sql_res

    rag_res = rag_agent(query)
    if rag_res:
        return rag_res

    return llm_fallback(query)

# =====================================================
# API ENDPOINTS
# =====================================================
@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest):
    return router_agent(req.query)