import os
import sys
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

# Tambah root project ke path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH    = os.path.join(PROJECT_ROOT, 'data', 'knowledge_data_indo.csv')
CHROMA_PATH  = os.path.join(PROJECT_ROOT, 'chroma_storage')

def init():
    print("Menghubungkan ke ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-mpnet-base-v2"
    )

    collection = client.get_or_create_collection(
        name="adaptive_interview_rag",
        embedding_function=embed_fn
    )

    if collection.count() > 0:
        print(f"ChromaDB sudah terisi ({collection.count()} dokumen). Skip.")
        return

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: File CSV tidak ditemukan di {DATA_PATH}")
        return

    print("Memproses CSV...")
    df = pd.read_csv(DATA_PATH).fillna("General Concept")

    collection.add(
        documents=df["Embedding_Text"].tolist(),
        metadatas=[
            {
                "role": str(row["Role"]),
                "stage": str(row["Stage"]),
                "difficulty_level": int(row["Adaptive_Level"]),
                "answer": str(row["Answer"])
            }
            for _, row in df.iterrows()
        ],
        ids=[f"q_{i}" for i in range(len(df))]
    )

    print(f"Selesai! {collection.count()} dokumen tersimpan.")

if __name__ == "__main__":
    init()