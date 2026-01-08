# =====================================================
# LOAD ENV (WAJIB PALING ATAS)
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
# ENV & CLIENTS
# =====================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY tidak ditemukan di environment variable")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

qdrant = None
if QDRANT_URL and QDRANT_API_KEY:
    qdrant = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=30.0
    )

DB_PATH = "olist.db"   # pastikan file ini ada di container

# =====================================================
# REQUEST MODEL
# =====================================================
class ChatRequest(BaseModel):
    query: str

# =====================================================
# UTILITIES
# =====================================================
def get_db():
    return sqlite3.connect(DB_PATH)

def embed_text(text: str):
    return openai_client.embed_
