import os

from dotenv import load_dotenv
from dotenv import load_dotenv
from groq import Groq
import pandas as pd
import json
from sentence_transformers import SentenceTransformer, util

# Load environment variables
load_dotenv()

ENV_MODE = os.getenv("ENV_MODE", "development")

embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

groq_client = Groq(
    api_key=os.environ.get("API_GROQ"),
)

def run_weight_experiment():
    # 1. Siapkan Dataset Skenario (Ground Truth)
    skenario_dataset = [
        {
            "nama_skenario": "Skenario A (Logika Benar tapi Bahasa Tidak Formal)",
            "question": "Jelaskan konsep dasar dari arsitektur Microservices.",
            "ideal_answer": "Microservices adalah pendekatan arsitektur di mana aplikasi dibangun sebagai kumpulan layanan kecil yang independen, berkomunikasi melalui API ringan, dan dapat di-deploy secara mandiri.",
            "user_answer": "Itu tuh cara bikin aplikasi yang dipecah jadi bagian-bagian kecil. Jadi kalau satu bagian rusak, yang lainnya tetep jalan. Terus mereka ngobrolnya pakai jalur komunikasi masing-masing.",
            "human_expert_score": 85 
        },
        {
            "nama_skenario": "Skenario B (Kata Kunci Ada tapi Tidak Memahami Konsep)",
            "question": "Jelaskan konsep dasar dari arsitektur Microservices.",
            "ideal_answer": "Microservices adalah pendekatan arsitektur di mana aplikasi dibangun sebagai kumpulan layanan kecil yang independen, berkomunikasi melalui API ringan, dan dapat di-deploy secara mandiri.",
            "user_answer": "Microservices adalah aplikasi, deploy, API ringan, dan independen. Kumpulan layanan ini sangat bagus untuk database SQL.",
            "human_expert_score": 30 
        },
        {
            "nama_skenario": "Skenario C (Jawaban Sempurna / Sesuai Textbook)",
            "question": "Jelaskan konsep dasar dari arsitektur Microservices.",
            "ideal_answer": "Microservices adalah pendekatan arsitektur di mana aplikasi dibangun sebagai kumpulan layanan kecil yang independen, berkomunikasi melalui API ringan, dan dapat di-deploy secara mandiri.",
            "user_answer": "Microservices merupakan arsitektur perangkat lunak yang memecah aplikasi besar menjadi sekumpulan layanan kecil yang berdiri sendiri atau independen. Setiap layanan ini biasanya berkomunikasi menggunakan API yang ringan dan dapat dikembangkan serta di-deploy secara terpisah.",
            "human_expert_score": 95 
        },
        {
            "nama_skenario": "Skenario D (Salah Total / Halusinasi)",
            "question": "Jelaskan konsep dasar dari arsitektur Microservices.",
            "ideal_answer": "Microservices adalah pendekatan arsitektur di mana aplikasi dibangun sebagai kumpulan layanan kecil yang independen, berkomunikasi melalui API ringan, dan dapat di-deploy secara mandiri.",
            "user_answer": "Microservices adalah jenis perangkat keras komputer berukuran sangat kecil buatan Microsoft yang dipasang di dalam motherboard untuk mempercepat kinerja prosesor dan RAM saat bermain game berat.",
            "human_expert_score": 10 
        }
    ]

    # Kombinasi bobot yang akan diuji: (Bobot LLM, Bobot Cosine)
    kombinasi_bobot = [(1.0, 0.0), (0.9, 0.1), (0.8, 0.2), (0.7, 0.3), (0.6, 0.4), (0.5, 0.5), (0.4, 0.6), (0.3, 0.7), (0.2, 0.8), (0.1, 0.9), (0.0, 1.0)]
    
    hasil_eksperimen = []

    print("[System] Memulai Eksperimen Pengujian Bobot...")

    # 2. Proses Setiap Skenario
    for data in skenario_dataset:
        print(f"\nMemproses: {data['nama_skenario']}")
        
        
        # Hitung Cosine asli
        ideal_emb = embedder.encode(data["ideal_answer"])
        user_emb = embedder.encode(data["user_answer"])
        cosine_raw = util.cos_sim(ideal_emb, user_emb).item()
        real_cosine_score = max(0, round(cosine_raw * 100))

        # Hitung LLM asli via Groq
        prompt = f"""Bertindak sebagai penilai interview teknis.
        Pertanyaan: {data['question']}
        Kunci Referensi: {data['ideal_answer']}
        Jawaban Kandidat: {data['user_answer']}
        Output format JSON: {{"score": 0-100, "feedback": "Kritik singkat"}}"""
        
        try:
            res = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}], temperature=0.2
            )
            content = res.choices[0].message.content.replace("```json","").replace("```","").strip()
            llm_result = json.loads(content)
            real_llm_score = int(llm_result.get("score", 50))
        except Exception as e:
            print(f"Error LLM: {e}")
            real_llm_score = 50

        print(f"  -> Raw LLM Score: {real_llm_score} | Raw Cosine Score: {real_cosine_score}")

        for w_llm, w_cos in kombinasi_bobot:
            final_score = round((real_llm_score * w_llm) + (real_cosine_score * w_cos))
            error = abs(data["human_expert_score"] - final_score)
            
            hasil_eksperimen.append({
                "Skenario": data["nama_skenario"],
                "Bobot LLM": f"{int(w_llm*100)}%",
                "Bobot Cosine": f"{int(w_cos*100)}%",
                "Raw LLM": real_llm_score,
                "Raw Cosine": real_cosine_score,
                "Final Score": final_score,
                "Skor Pakar": data["human_expert_score"],
                "Error": error
            })

    # 3. Export Hasil ke DataFrame untuk Laporan Skripsi
    df_hasil = pd.DataFrame(hasil_eksperimen)
    df_hasil.to_csv("hasil_eksperimen_bobot.csv", index=False)
    print("\n[System] Eksperimen Selesai. Hasil disimpan ke 'hasil_eksperimen_bobot.csv'")
    print(df_hasil.to_markdown(index=False))

# Jalankan eksperimen
if __name__ == "__main__":
    run_weight_experiment()