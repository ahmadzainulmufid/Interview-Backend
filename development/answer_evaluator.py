import os
from groq import Groq
from dotenv import load_dotenv
from development.cosine_evaluator import cosine_sentence_analysis, compute_cosine_score

load_dotenv()

THRESHOLD_GOOD = 0.5
THRESHOLD_AVERAGE = 0.05


def get_groq_client():
    """Inisialisasi client Groq untuk evaluator"""
    api_key = os.getenv("TTS_API_KEY")
    if not api_key:
        raise ValueError("ERROR: API_KEY Groq belum diset di file .env")
    return Groq(api_key=api_key)


def evaluate_answer_ai(question, ideal_answer, user_answer):
    word_count = len(user_answer.strip().split())
    if word_count < 10:
        print(f"[TFIDF EVAL] Jawaban terlalu pendek ({word_count} kata) → Score: 0")
        return {
            "score": 0,
            "feedback": "Jawaban terlalu singkat untuk dievaluasi. Harap berikan penjelasan yang lebih lengkap.",
            "status": "Bad",
            "cosine_detail": {
                "keyword_coverage_pct": 0.0,
                "keywords_found": [],
                "keywords_missing": [],
                "tfidf_cosine_score": 0.0,
            }
        }

    cosine_data = cosine_sentence_analysis(ideal_answer, user_answer)
    final_score = cosine_data["tfidf_cosine_score"]  
    if final_score >= THRESHOLD_GOOD:
        status = "Good"
    elif final_score > THRESHOLD_AVERAGE:
        status = "Average"
    else:
        status = "Bad"

    keywords_found_str = ", ".join(cosine_data["keywords_found"]) if cosine_data["keywords_found"] else "Tidak ada"
    keywords_missing_str = ", ".join(cosine_data["missing_keywords"]) if cosine_data["missing_keywords"] else "Tidak ada"

    sorted_sentences = sorted(cosine_data["sentence_scores"], key=lambda x: x["best_similarity"])
    weak_sentences = sorted_sentences[:2]
    weak_str = "\n".join(
        f'  - "{s["sentence"]}" (sim={s["best_similarity"]:.2f}, '
        f'mirip dengan: "{s["matched_ideal"]}")'
        for s in weak_sentences
    )

    client = get_groq_client()

    system_instruction = f"""Kamu adalah Senior IT HRD yang menilai jawaban kandidat interview secara konsisten dan terstruktur.

    <data_analisis>
    Pertanyaan   : {question}
    Skor akhir   : {final_score:.0%} (Status: {status})
    Keyword ditemukan   : [{keywords_found_str}]
    Keyword tidak muncul: [{keywords_missing_str}]
    Kalimat kandidat dengan relevansi terendah:
    {weak_str}
    </data_analisis>
    
    Tulis feedback PERSIS 2 kalimat, format tetap:
    1. Good/Average → "Jawaban Anda sudah mencakup [konsep yang benar], namun belum membahas [1-2 konsep teknis yang hilang]." | Bad → "Jawaban Anda belum membahas [1-2 konsep teknis yang hilang]."
    2. "Untuk memperkuat jawaban, [saran konkret + contoh singkat]."
    
    Aturan: konsep yang disebut harus istilah teknis bermakna (bukan kata umum/bilangan seperti "tiga"); jangan sebut angka skor; fokus konten teknis bukan tata bahasa. Output HANYA 2 kalimat itu, tanpa preamble/markdown."""
    feedback = "Evaluasi berhasil dilakukan."

    try:
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": system_instruction}],
            temperature=0.3
        )
        feedback = res.choices[0].message.content.strip()
    except Exception as e:
        print(f"[LLM FEEDBACK] Error: {e}")

    print(
        f"[TFIDF EVAL] KW={cosine_data['keyword_coverage']:.0%} | "
        f"TFIDF-Cosine={cosine_data['tfidf_cosine_score']:.2f} | "
        f"Final={final_score:.0%}"
    )

    return {
        "score": round(final_score * 100),
        "feedback": feedback,
        "status": status,
        "cosine_detail": {
            "keyword_coverage_pct": float(round(cosine_data["keyword_coverage"] * 100, 1)),
            "keywords_found": cosine_data["keywords_found"],
            "keywords_missing": cosine_data["missing_keywords"],
            "tfidf_cosine_score": float(cosine_data["tfidf_cosine_score"]),
        }
    }