import json
import re
from flask.cli import load_dotenv
import pandas as pd
from tqdm import tqdm
import os
from groq import Groq

load_dotenv()

ENV_MODE = os.getenv("ENV_MODE", "development")

groq_client = Groq(
    api_key=os.environ.get("API_GROQ"),
)

def safe_json_load(content):
    """Fungsi untuk membersihkan dan mengekstrak JSON dari respon LLM"""
    content = re.sub(r"```json|```", "", content).strip()
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if match:
        content = match.group(0)
    try:
        return json.loads(content)
    except Exception as e:
        print(f"\n[DEBUG Error JSON] Raw LLM Output: {content}")
        return None

def evaluate_context_precision(query_text, q_text, ans_text):
    system_instruction = f"""Tugas Anda adalah menilai 'Context Precision'.
Apakah data referensi (Data Terambil) relevan dan berguna untuk menjawab topik pencarian?
Topik Pencarian: {query_text}
Data Terambil: {q_text} | {ans_text}

Instruksi: Berikan skor antara 0.0000 hingga 1.0000. (Gunakan desimal spesifik, contoh: 0.8132, 0.9200, 0.4550).
Output HANYA dalam format JSON strict:
{{"precision_score": <skor>}}"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile", 
        messages=[{"role": "user", "content": system_instruction}], 
        temperature=0.1
    )
    res = safe_json_load(response.choices[0].message.content)
    return float(res.get("precision_score", 0.0)) if res else 0.0

def evaluate_faithfulness(q_text, context_doc):
    system_instruction = f"""Tugas Anda adalah menilai 'Faithfulness'.
Apakah pertanyaan yang dihasilkan sepenuhnya berdasarkan pada konteks referensi, tanpa mengarang informasi di luar konteks?
Konteks Referensi (RAG): {context_doc}
Pertanyaan AI: {q_text}

Instruksi: Berikan skor antara 0.0000 hingga 1.0000. (Gunakan desimal spesifik, contoh: 0.8132, 0.9200, 0.4550).
Output HANYA dalam format JSON strict:
{{"faithfulness_score": <skor>}}"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile", 
        messages=[{"role": "user", "content": system_instruction}], 
        temperature=0.1
    )
    res = safe_json_load(response.choices[0].message.content)
    return float(res.get("faithfulness_score", 0.0)) if res else 0.0

def evaluate_answer_relevance(q_text, role):
    system_instruction = f"""Tugas Anda adalah menilai 'Question Relevance'.
Apakah pertanyaan ini valid, masuk akal, dan berbobot untuk ditanyakan kepada kandidat pada profesi berikut?
Role: {role}
Pertanyaan: {q_text}

Instruksi: Berikan skor antara 0.0000 hingga 1.0000. (Gunakan desimal spesifik, contoh: 0.8132, 0.9200, 0.4550).
Output HANYA dalam format JSON strict:
{{"relevance_score": <skor>}}"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile", 
        messages=[{"role": "user", "content": system_instruction}], 
        temperature=0.1
    )
    res = safe_json_load(response.choices[0].message.content)
    return float(res.get("relevance_score", 0.0)) if res else 0.0

def run_rag_evaluation(csv_path="D:\\interview-backend\\data\\hasil_interview_history.csv"):
    print(f"\nMembaca data history wawancara dari '{csv_path}'...\n")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"File '{csv_path}' tidak ditemukan!")
        return

    target_role = input("Masukkan Posisi/Role (contoh: Backend Engineer): ") or "Backend Engineer"
    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Mengevaluasi Kinerja RAG"):
        if pd.isna(row.get('retrieved_query_text')) or pd.isna(row.get('retrieved_context_doc')):
            continue 

        precision = evaluate_context_precision(row['retrieved_query_text'], row['retrieved_context_doc'], row['ideal_answer'])
        faithfulness = evaluate_faithfulness(row['question'], row['retrieved_context_doc'])
        relevance = evaluate_answer_relevance(row['question'], target_role)

        results.append({
            "stage": row.get('stage', 'Unknown'),
            "context_precision": precision,
            "faithfulness": faithfulness,
            "answer_relevance": relevance
        })

    if not results:
        print("\nTidak ada data berbasis RAG yang dievaluasi.")
        return

    eval_df = pd.DataFrame(results)
    
    metrics_summary = {
        "context_precision": eval_df['context_precision'].mean(),
        "faithfulness": eval_df['faithfulness'].mean(),
        "answer_relevance": eval_df['answer_relevance'].mean()
    }

    print("\n" + "="*50)
    print(" HASIL EVALUASI RAGAS")
    print("="*50)
    print(f" Context Precision : {metrics_summary['context_precision']:.4f}")
    print(f" Faithfulness      : {metrics_summary['faithfulness']:.4f}")
    print(f" Answer Relevance  : {metrics_summary['answer_relevance']:.4f}")
    print("="*50)

    export_filename = "ragas_evaluation_results.csv"
    eval_df.to_csv(export_filename, index=False, float_format='%.4f')
    print(f"\nFile data mentah evaluasi berhasil disimpan ke '{export_filename}'")

if __name__ == "__main__":
    run_rag_evaluation()