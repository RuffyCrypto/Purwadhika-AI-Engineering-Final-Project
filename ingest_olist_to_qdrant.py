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
    api_key=os.getenv("QDRANT_API_KEY")
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
# PARAMETER INGEST
# =========================
MAX_ROWS = 300          # total data
BATCH_SIZE = 25         # ⬅️ INI KUNCI (jangan besar)
SLEEP_SECONDS = 1       # jeda antar batch

df = df.head(MAX_ROWS)

# =========================
# INGEST BERTAHAP
# =========================
points_batch = []
uploaded = 0

for idx, row in df.iterrows():
    document = f"""
ID Produk: {row['product_id']}
Kategori Produk: {row['product_category_name']}
Harga Produk: {row['price']}
Kota Seller: {row['seller_city']}
Provinsi Seller: {row['seller_state']}
Review Pelanggan: {row['review_comment_message']}
Rating: {row['review_score']}
"""

    embedding = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=document
    ).data[0].embedding

    points_batch.append({
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

    # =========================
    # UPLOAD PER BATCH
    # =========================
    if len(points_batch) >= BATCH_SIZE:
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=points_batch
        )
        uploaded += len(points_batch)
        print(f"✅ Uploaded {uploaded}/{len(df)}")
        points_batch = []
        time.sleep(SLEEP_SECONDS)

# =========================
# SISA DATA
# =========================
if points_batch:
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=points_batch
    )
    uploaded += len(points_batch)

print(f"🎉 INGEST SELESAI: {uploaded} dokumen masuk ke Qdrant")