import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

collection = None

def init_chroma_db(csv_path="/content/drive/MyDrive/Skripsi/Dataset/knowledge_data_rag_adaptive_final.csv"):
    global collection
    
    knowledge_df = pd.read_csv(csv_path).fillna("General Concept")
    
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    chroma_client = chromadb.Client()
    
    try:
        chroma_client.delete_collection(name="adaptive_interview_rag")
    except:
        pass
        
    collection = chroma_client.create_collection(
        name="adaptive_interview_rag",
        embedding_function=embed_fn
    )
    
    documents = knowledge_df["Embedding_Text"].tolist()
    ids = [f"q_{i}" for i in range(len(knowledge_df))]
    
    metadatas = []
    for _, row in knowledge_df.iterrows():
        metadatas.append({
            "role": str(row["Role"]),
            "stage": str(row["Stage"]),
            "difficulty_level": int(row["Adaptive_Level"]),
            "answer": str(row["Answer"])
        })
        
    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )
    print(f"✅ Vector DB Ready. Loaded {len(knowledge_df)} records.")
    return collection

def get_collection():
    return collection