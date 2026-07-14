"""
Perhitungan Skor Akhir Jawaban Kandidat
========================================
Komponen:
  1. Keyword Coverage  (bobot 30%) - TF-IDF + Stopwords Indonesia
  2. Sentence Cosine   (bobot 40%) - Embedding per kalimat
  3. Semantic Coverage (bobot 30%) - Embedding keseluruhan teks

Model: paraphrase-multilingual-mpnet-base-v2
"""

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk

# Download tokenizer kalimat jika belum ada
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── Konstanta bobot ──────────────────────────────────────────────────────────
WEIGHT_KEYWORD  = 0.30
WEIGHT_SENTENCE = 0.40
WEIGHT_SEMANTIC = 0.30

THRESHOLD_GOOD    = 75
THRESHOLD_AVERAGE = 50

# ── Stopwords Indonesia ──────────────────────────────────────────────────────
# Kata-kata umum yang tidak bermakna sebagai kata kunci teknis
STOPWORDS_ID = {
    "dan", "atau", "yang", "di", "ke", "dari", "dengan", "untuk", "pada",
    "adalah", "ini", "itu", "juga", "lebih", "dapat", "bisa", "ada",
    "tidak", "dalam", "oleh", "sebagai", "akan", "telah", "sudah",
    "antara", "serta", "sehingga", "namun", "karena", "jika", "ketika",
    "agar", "seperti", "tanpa", "secara", "setiap", "hanya", "kita",
    "kami", "mereka", "nya", "pun", "lah", "kah", "pula", "maupun",
    "bahwa", "sedangkan", "melalui", "terhadap", "saat", "hal", "cara",
    "jenis", "berbagai", "beberapa", "semua", "sebuah", "suatu",
    "tersebut", "masing", "hingga", "digunakan", "memerlukan",
    "merupakan", "mempermudah", "memungkinkan", "meningkatkan",
}

# ── Model embedding ──────────────────────────────────────────────────────────
print("Memuat model embedding...")
model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
print("Model berhasil dimuat.\n")


# ── Helper: tokenisasi kalimat ───────────────────────────────────────────────
def split_sentences(text: str) -> list[str]:
    """Memecah teks menjadi daftar kalimat."""
    sentences = nltk.sent_tokenize(text)
    return [s.strip() for s in sentences if s.strip()]


# ── 1. Keyword Coverage (TF-IDF + Stopwords) ─────────────────────────────────
def hitung_keyword_coverage(jawaban_referensi: str, jawaban_kandidat: str) -> tuple[float, list, list]:
    """
    Mengekstrak kata kunci bermakna dari referensi menggunakan TF-IDF,
    memfilter stopwords Indonesia, lalu menghitung proporsi kata kunci
    yang muncul di jawaban kandidat.

    Returns:
        coverage (float) : nilai 0.0 – 1.0
        found    (list)  : kata kunci yang ditemukan
        missing  (list)  : kata kunci yang tidak ditemukan
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words=None,
        sublinear_tf=True,
        max_features=200,
        token_pattern=r"(?u)\b[a-zA-Z]\w{2,}\b",  # minimal 3 karakter
    )
    tfidf_matrix  = vectorizer.fit_transform([jawaban_referensi])
    feature_names = vectorizer.get_feature_names_out()
    scores        = tfidf_matrix.toarray()[0]

    # Urutkan berdasarkan skor TF-IDF tertinggi, filter stopwords
    ranked = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)

    top_n = 8
    kata_kunci = []
    for term, score in ranked:
        if score <= 0:
            continue
        # Buang term jika semua tokennya adalah stopword
        tokens = term.split()
        if all(t.lower() in STOPWORDS_ID for t in tokens):
            continue
        kata_kunci.append(term)
        if len(kata_kunci) >= top_n:
            break

    # Cek keberadaan setiap kata kunci di jawaban kandidat
    kandidat_lower = jawaban_kandidat.lower()
    found   = [kw for kw in kata_kunci if kw.lower() in kandidat_lower]
    missing = [kw for kw in kata_kunci if kw.lower() not in kandidat_lower]

    coverage = len(found) / len(kata_kunci) if kata_kunci else 0.0
    return coverage, found, missing


# ── 2. Sentence Cosine (Embedding per kalimat) ───────────────────────────────
def hitung_sentence_cosine(jawaban_referensi: str, jawaban_kandidat: str) -> float:
    """
    Untuk setiap kalimat kandidat, cari nilai cosine similarity tertinggi
    terhadap semua kalimat referensi, lalu rata-ratakan.

    Returns:
        rata-rata nilai cosine tertinggi (float 0.0 – 1.0)
    """
    kalimat_ref      = split_sentences(jawaban_referensi)
    kalimat_kandidat = split_sentences(jawaban_kandidat)

    if not kalimat_ref or not kalimat_kandidat:
        return 0.0

    emb_ref      = model.encode(kalimat_ref,      convert_to_numpy=True)
    emb_kandidat = model.encode(kalimat_kandidat, convert_to_numpy=True)

    sim_matrix      = cosine_similarity(emb_kandidat, emb_ref)
    max_per_kalimat = sim_matrix.max(axis=1)
    return float(max_per_kalimat.mean())


# ── 3. Semantic Coverage (Embedding keseluruhan teks) ────────────────────────
def hitung_semantic_coverage(jawaban_referensi: str, jawaban_kandidat: str) -> float:
    """
    Embedding seluruh jawaban kandidat dibandingkan dengan setiap
    kalimat referensi, lalu rata-ratakan nilai cosine-nya.

    Returns:
        rata-rata nilai cosine (float 0.0 – 1.0)
    """
    kalimat_ref = split_sentences(jawaban_referensi)

    if not kalimat_ref:
        return 0.0

    emb_ref      = model.encode(kalimat_ref,        convert_to_numpy=True)
    emb_kandidat = model.encode([jawaban_kandidat],  convert_to_numpy=True)

    sim_scores = cosine_similarity(emb_kandidat, emb_ref)[0]
    return float(sim_scores.mean())


# ── 4. Weighted Linear Combination → Skor Akhir ──────────────────────────────
def hitung_skor_akhir(keyword_cov: float, sentence_cos: float, semantic_cov: float) -> float:
    """
    Skor Akhir = (0.30 × Keyword Coverage)
               + (0.40 × Sentence Cosine)
               + (0.30 × Semantic Coverage)
    Hasil dikalikan 100 → skala 0–100.
    """
    skor = (WEIGHT_KEYWORD  * keyword_cov
          + WEIGHT_SENTENCE * sentence_cos
          + WEIGHT_SEMANTIC * semantic_cov)
    return round(skor * 100, 2)


def tentukan_status(skor: float) -> str:
    if skor >= THRESHOLD_GOOD:
        return "Good"
    elif skor >= THRESHOLD_AVERAGE:
        return "Average"
    else:
        return "Bad"


# ── Pipeline utama ────────────────────────────────────────────────────────────
def evaluasi_jawaban(pertanyaan: str, jawaban_referensi: str, jawaban_kandidat: str) -> dict:
    """Menjalankan seluruh pipeline penilaian dan mengembalikan dict hasil."""
    print(f"\n{'='*65}")
    print(f"Pertanyaan : {pertanyaan[:80]}...")
    print(f"{'='*65}")

    kw_cov, found, missing = hitung_keyword_coverage(jawaban_referensi, jawaban_kandidat)
    sent_cos = hitung_sentence_cosine(jawaban_referensi, jawaban_kandidat)
    sem_cov  = hitung_semantic_coverage(jawaban_referensi, jawaban_kandidat)
    skor     = hitung_skor_akhir(kw_cov, sent_cos, sem_cov)
    status   = tentukan_status(skor)

    print(f"\n[1] Keyword Coverage  : {kw_cov*100:.2f}%")
    print(f"    Ditemukan  : {found}")
    print(f"    Tidak ada  : {missing}")
    print(f"\n[2] Sentence Cosine   : {sent_cos:.4f}")
    print(f"\n[3] Semantic Coverage : {sem_cov:.4f}")
    print(f"\n--- Weighted Linear Combination ---")
    print(f"    (0.30 × {kw_cov:.4f}) + (0.40 × {sent_cos:.4f}) + (0.30 × {sem_cov:.4f})")
    print(f"  = {WEIGHT_KEYWORD*kw_cov:.4f} + {WEIGHT_SENTENCE*sent_cos:.4f} + {WEIGHT_SEMANTIC*sem_cov:.4f}")
    print(f"  = {(WEIGHT_KEYWORD*kw_cov + WEIGHT_SENTENCE*sent_cos + WEIGHT_SEMANTIC*sem_cov):.4f} × 100")
    print(f"\n>>> SKOR AKHIR : {skor}  →  Status: {status}")

    return {
        "pertanyaan"      : pertanyaan,
        "keyword_coverage": round(kw_cov * 100, 2),
        "keywords_found"  : found,
        "keywords_missing": missing,
        "sentence_cosine" : round(sent_cos, 4),
        "semantic_coverage": round(sem_cov, 4),
        "skor_akhir"      : skor,
        "status"          : status,
    }


# ── Data uji (2 skenario dari dokumen) ───────────────────────────────────────
if __name__ == "__main__":

    skenario = [
        {
            "pertanyaan": (
                "Apa yang membuat antarmuka (interface) menjadi komponen penting "
                "dalam pemrograman, dan bagaimana hal ini memengaruhi kemudahan "
                "perawatan serta pengembangan sistem yang kompleks?"
            ),
            "jawaban_referensi": (
                "Antarmuka penting dalam pemrograman karena memungkinkan abstraksi "
                "dan modularitas, memisahkan antarmuka dari implementasi, sehingga "
                "meningkatkan kemudahan dan kemudahan perawatan kode. Antarmuka juga "
                "membantu mengurangi ketergantungan antar komponen dan meningkatkan "
                "penggunaan kembali kode. Dengan demikian, antarmuka mempermudah "
                "pengembangan dan pemeliharaan sistem yang kompleks."
            ),
            "jawaban_kandidat": (
                "Baik, untuk interface merupakan sebuah kontrak formal dalam kode "
                "yang menentukan apa fungsi yang harus disediakan oleh sebuah komponen "
                "tanpa bagaimana caranya komponen tersebut dijalankan."
            ),
        },
        {
            "pertanyaan": (
                "Kapan kita menggunakan list daripada antarmuka dalam pemrograman, "
                "dan apa kelebihan serta kekurangan dari masing-masing pilihan tersebut?"
            ),
            "jawaban_referensi": (
                "Kita menggunakan list ketika kita memerlukan akses elemen secara "
                "langsung melalui indeks, sedangkan antarmuka lebih fleksibel dan dapat "
                "diimplementasikan oleh berbagai jenis koleksi. List juga lebih cocok "
                "ketika kita memerlukan operasi seperti penyisipan, penghapusan, atau "
                "pengeditan elemen pada posisi tertentu. Antarmuka seperti ICollection "
                "atau IEnumerable lebih umum dan dapat digunakan ketika kita hanya "
                "memerlukan iterasi atas koleksi tanpa memperhatikan jenis koleksi "
                "yang digunakan."
            ),
            "jawaban_kandidat": (
                "Untuk dalam pemrograman, Pembandingkan List dan Antarmuka atau Interface, "
                "Sebenarnya, yaitu perbandingan keduanya, interface ini merupakan sebuah "
                "konsep struktur cetak yang berupa abstract. Kalau list ini sebuah struktur "
                "data yang concrete, sehingga implementasi nyata untuk menyimpan kumpulan data."
            ),
        },
    ]

    hasil_semua = []
    for s in skenario:
        hasil = evaluasi_jawaban(s["pertanyaan"], s["jawaban_referensi"], s["jawaban_kandidat"])
        hasil_semua.append(hasil)

    print(f"\n\n{'='*65}")
    print("RINGKASAN HASIL")
    print(f"{'='*65}")
    for i, h in enumerate(hasil_semua, 1):
        print(f"Skenario {i}: Skor={h['skor_akhir']:5.2f}  Status={h['status']}")