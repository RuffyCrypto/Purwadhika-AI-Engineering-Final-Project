from dotenv import load_dotenv
load_dotenv()

import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

COLLECTION_NAME = "olist_products"
VECTOR_SIZE = 1536  # ukuran embedding OpenAI text-embedding-3-small

# Hapus collection jika sudah ada (biar aman saat dev)
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

# Buat collection baru
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=VECTOR_SIZE,
        distance=Distance.COSINE
    )
)

print(f"✅ Collection '{COLLECTION_NAME}' berhasil dibuat")