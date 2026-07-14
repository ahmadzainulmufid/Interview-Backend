import os
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def get_groq_client():
    """Inisialisasi client Groq untuk generator laporan"""
    api_key = os.getenv("API_GROQ") 
    if not api_key:
        raise ValueError("ERROR: API_KEY Groq belum diset di file .env")
    return Groq(api_key=api_key)

def generate_final_report_ai(history, role):
    client = get_groq_client()
    blocks = []
    
    for i, h in enumerate(history, 1):
        block = (
            f"[Pertanyaan {i}] Stage: {h.get('stage', '-')}\n"
            f"  Q : {h['question']}\n"
            f"  A : {h['user_answer']}"
        )
        blocks.append(block)
        
    data_str = "\n\n".join(blocks)
    
    system_instruction = f"""Buatlah ringkasan evaluasi wawancara kerja untuk kandidat posisi {role} berdasarkan data wawancara berikut:

{data_str}

Buatlah ringkasan evaluasi wawancara dalam Bahasa Indonesia, hanya berdasarkan isi jawaban kandidat di atas.

Hasilkan output dengan struktur HANYA seperti di bawah ini, tanpa preamble atau penutup tambahan:

[Paragraf 1: Ringkasan substansi wawancara secara keseluruhan. Rangkum topik-topik dan poin-poin utama yang dibahas kandidat di sepanjang sesi wawancara (dari opening, technical, hingga case study), secara netral dan deskriptif. Sebutkan apa yang disampaikan kandidat, bukan seberapa baik atau buruk itu.]
 
[Paragraf 2: Berisi 2-3 kalimat yang menilai kelayakan teknis kandidat (pemahaman tugas pokok/domain posisi {role}) berdasarkan logika dan isi jawabannya]
 
[Paragraf 3: Berisi 2-3 kalimat yang menilai struktur berpikir kandidat (bagaimana ia menyusun alur jawabannya, runtut atau melompat-lompat) berdasarkan logika penjelasannya]
"""
    try:
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": system_instruction}],
            temperature=0.2
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        print(f"[FINAL REPORT] Error: {e}")
        return f"Gagal menghasilkan laporan: {e}"