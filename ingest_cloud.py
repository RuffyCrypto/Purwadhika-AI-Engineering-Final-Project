from dotenv import load_dotenv
load_dotenv()

import os
import pandas as pd
import time
from openai import OpenAI
from qdrant_client import QdrantClient

# =========================
# INIT CLIENTS
# =========================
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=120.0   # cloud aman pakai besar
)

COLLECTION_NAME = "olist_products"

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv(
    "combined_olist_datasetclean.csv",
    low_memory=False
)

# =========================
# SELECT & CLEAN
# =========================
df = df[[
    "product_id",
    "product_category_name",
    "price",
    "seller_city",
    "seller_state",
    "review_comment_message",
    "review_score"
]]

df = df.dropna(subset=["product_id", "product_category_name"])

df["seller_city"] = df["seller_city"].fillna("unknown")
df["seller_state"] = df["seller_state"].fillna("unknown")
df["price"] = df["price"].fillna(0)
df["review_comment_message"] = df["review_comment_message"].fillna("")
df["review_score"] = df["review_score"].fillna(0)

# =========================
# LIMIT (CUKUP UNTUK CAPSTONE)
# =========================
MAX_ROWS = 100   # 🔥 cukup untuk demo & sidang
BATCH_SIZE = 20

df = df.head(MAX_ROWS)

# =========================
# INGEST
# =========================
points = []
uploaded = 0

for idx, row in df.iterrows():
    review_text = str(row["review_comment_message"])[:500]

    document = f"""
ID Produk: {row['product_id']}
Kategori Produk: {row['product_category_name']}
Harga Produk: {row['price']}
Kota Seller: {row['seller_city']}
Provinsi Seller: {row['seller_state']}
Review Pelanggan: {review_text}
Rating: {row['review_score']}
"""

    embedding = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=document
    ).data[0].embedding

    points.append({
        "id": int(idx),
        "vector": embedding,
        "payload": {
            "text": document,
            "category": row["product_category_name"],
            "seller_city": row["seller_city"],
            "price": float(row["price"]),
            "review_score": float(row["review_score"])
        }
    })

    if len(points) >= BATCH_SIZE:
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        uploaded += len(points)
        print(f"Uploaded {uploaded}/{len(df)}")
        points = []

# sisa
if points:
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    uploaded += len(points)

print(f"✅ INGEST SELESAI: {uploaded} dokumen")