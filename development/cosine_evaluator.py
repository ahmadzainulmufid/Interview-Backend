import math
import re
import json
import os
from collections import Counter

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    _stemmer = StemmerFactory().create_stemmer()
except ImportError:
    _stemmer = None
    print("Library 'Sastrawi' belum terinstall.")


_stem_cache: dict[str, str] = {}

STOPWORDS_ID = {
    "dan", "atau", "yang", "di", "ke", "dari", "dengan", "untuk", "pada",
    "adalah", "ini", "itu", "juga", "lebih", "dapat", "bisa", "ada",
    "tidak", "dalam", "oleh", "sebagai", "akan", "telah", "sudah",
    "antara", "serta", "sehingga", "namun", "karena", "jika", "ketika",
    "agar", "seperti", "tanpa", "secara", "setiap", "hanya", "kita",
    "kami", "mereka", "nya", "pun", "lah", "kah", "pula",
    "maupun", "bahwa", "sedangkan", "melalui", "terhadap", "saat",
    "hal", "cara", "jenis", "berbagai", "beberapa", "semua", "sebuah",
    "suatu", "tersebut", "masing", "hingga", "digunakan",
    "memerlukan", "merupakan", "mempermudah", "memungkinkan",
    "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh",
    "delapan", "sembilan", "sepuluh", "pertama", "kedua", "ketiga",
}

SYNONYM_MAP = {
    "chap": "cap",
    "ketersediaan": "availability",
    "konsistensi": "consistency",
    "toleransi partisi": "partition tolerance",
    "layanan mikro": "microservice",
    "microservices": "microservice",
    "basis data": "database",
    "antrian pesan": "message queue",
    "penyeimbang beban": "load balancer",
}

TOKEN_PATTERN = re.compile(r"[a-z]{3,}")


def normalize_synonyms(text: str) -> str:
    lowered = text.lower()
    for source, target in SYNONYM_MAP.items():
        lowered = re.sub(rf"\b{re.escape(source)}\b", target, lowered)
    return lowered


def case_folding(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenizing(folded_text: str) -> list[str]:
    return TOKEN_PATTERN.findall(folded_text)


def filtering(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in STOPWORDS_ID]


def stemming(tokens: list[str]) -> list[str]:
    if _stemmer is None:
        return tokens
    stemmed_tokens = []
    for token in tokens:
        if token not in _stem_cache:
            _stem_cache[token] = _stemmer.stem(token)
        stemmed_tokens.append(_stem_cache[token])
    return stemmed_tokens


def preprocess_text(text: str) -> list[str]:
    normalized = normalize_synonyms(text)
    folded = case_folding(normalized)
    tokens = tokenizing(folded)
    tokens = filtering(tokens)
    tokens = stemming(tokens)
    tokens = [t for t in tokens if t and t not in STOPWORDS_ID]
    return tokens


def split_into_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    return sentences if sentences else [text.strip()]


def compute_tf(tokens: list[str]) -> dict:
    if not tokens:
        return {}
    counts = Counter(tokens)
    return {term: 1.0 + math.log(count) for term, count in counts.items()}


def compute_idf(corpus_tokens: list[list[str]]) -> dict:
    N = len(corpus_tokens)
    df = Counter()
    for tokens in corpus_tokens:
        for term in set(tokens):
            df[term] += 1

    idf_dict = {}
    for term, freq in df.items():
        idf_dict[term] = math.log((1.0 + N) / (1.0 + freq)) + 1.0
    return idf_dict


def compute_tfidf_vector(tf: dict, idf: dict, vocab: list[str], default_idf: float = 1.0) -> list[float]:
    return [tf.get(term, 0.0) * idf.get(term, default_idf) for term in vocab]


def calculate_cosine_similarity(vector_a, vector_b) -> float:
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    magnitude_a = math.sqrt(sum(a ** 2 for a in vector_a))
    magnitude_b = math.sqrt(sum(b ** 2 for b in vector_b))
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    return dot_product / (magnitude_a * magnitude_b)

_GLOBAL_IDF: dict | None = None
_DEFAULT_IDF: float = 1.0
_CORPUS_SIZE: int = 0

_CACHE_PATH = os.path.join(os.path.dirname(__file__), "idf_cache.json")


def build_global_idf(all_ideal_answers: list[str], use_cache: bool = True) -> None:
    global _GLOBAL_IDF, _DEFAULT_IDF, _CORPUS_SIZE

    if use_cache and os.path.exists(_CACHE_PATH):
        with open(_CACHE_PATH, "r", encoding="utf-8") as f:
            cached = json.load(f)
        _GLOBAL_IDF = cached["idf"]
        _DEFAULT_IDF = cached["default_idf"]
        _CORPUS_SIZE = cached["corpus_size"]
        print(f"[IDF] Dimuat dari cache: {_CORPUS_SIZE} dokumen, {len(_GLOBAL_IDF)} term.")
        return

    corpus_tokens = [preprocess_text(ans) for ans in all_ideal_answers if ans and ans.strip()]
    _GLOBAL_IDF = compute_idf(corpus_tokens)
    _CORPUS_SIZE = len(corpus_tokens)
    _DEFAULT_IDF = math.log((1.0 + _CORPUS_SIZE) / (1.0 + 1)) + 1.0

    if use_cache:
        with open(_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {"idf": _GLOBAL_IDF, "default_idf": _DEFAULT_IDF, "corpus_size": _CORPUS_SIZE},
                f, ensure_ascii=False,
            )

    print(f"[IDF] Dibangun dari {_CORPUS_SIZE} jawaban ideal, {len(_GLOBAL_IDF)} term unik.")


def get_global_idf() -> tuple[dict, float]:
    if _GLOBAL_IDF is None:
        raise RuntimeError(
            "Global IDF belum di-build. Panggil build_global_idf(...) saat startup aplikasi "
            "sebelum memanggil fungsi evaluasi manapun."
        )
    return _GLOBAL_IDF, _DEFAULT_IDF


def hitung_tfidf_cosine(jawaban_referensi: str, jawaban_kandidat: str) -> tuple[float, dict]:
    tokens_ref = preprocess_text(jawaban_referensi)
    tokens_kandidat = preprocess_text(jawaban_kandidat)

    idf, default_idf = get_global_idf()

    tf_ref = compute_tf(tokens_ref)
    tf_kandidat = compute_tf(tokens_kandidat)

    vocab = sorted(set(tokens_ref) | set(tokens_kandidat))

    vector_ref = compute_tfidf_vector(tf_ref, idf, vocab, default_idf)
    vector_kandidat = compute_tfidf_vector(tf_kandidat, idf, vocab, default_idf)

    similarity = max(0.0, calculate_cosine_similarity(vector_ref, vector_kandidat))

    detail = {
        "vocab_size": len(vocab),
        "tokens_referensi": tokens_ref,
        "tokens_kandidat": tokens_kandidat,
        "tf_referensi": tf_ref,
        "tf_kandidat": tf_kandidat,
    }
    return similarity, detail


def extract_keywords_tfidf(tf_ref: dict, idf: dict, tokens_ref: list[str], default_idf: float, top_n: int = 8) -> list[str]:
    try:
        vocab_ref = set(tokens_ref)
        weighted = [(term, tf_ref.get(term, 0.0) * idf.get(term, default_idf)) for term in vocab_ref]
        weighted.sort(key=lambda x: x[1], reverse=True)
        return [term for term, score in weighted if score > 0][:top_n]
    except Exception as e:
        print(f"[KEYWORD EXTRACT] Error: {e}")
        return []


def hitung_keyword_coverage(jawaban_referensi: str, jawaban_kandidat: str, top_n: int = 8) -> tuple[float, list, list]:
    tokens_ref = preprocess_text(jawaban_referensi)
    tokens_kandidat = preprocess_text(jawaban_kandidat)

    idf, default_idf = get_global_idf()
    tf_ref = compute_tf(tokens_ref)

    keywords = extract_keywords_tfidf(tf_ref, idf, tokens_ref, default_idf, top_n=top_n)
    kandidat_token_set = set(tokens_kandidat)

    found = [kw for kw in keywords if kw in kandidat_token_set]
    missing = [kw for kw in keywords if kw not in kandidat_token_set]

    coverage = len(found) / len(keywords) if keywords else 0.0
    return coverage, found, missing


def hitung_sentence_tfidf_cosine(jawaban_referensi: str, jawaban_kandidat: str) -> tuple[float, list]:
    ideal_sentences = split_into_sentences(jawaban_referensi)
    user_sentences = split_into_sentences(jawaban_kandidat)

    ideal_tokens = [preprocess_text(s) for s in ideal_sentences]
    user_tokens = [preprocess_text(s) for s in user_sentences]

    idf, default_idf = get_global_idf()
    vocab = sorted(set(t for tokens in (ideal_tokens + user_tokens) for t in tokens))

    ideal_vectors = [compute_tfidf_vector(compute_tf(tokens), idf, vocab, default_idf) for tokens in ideal_tokens]
    user_vectors = [compute_tfidf_vector(compute_tf(tokens), idf, vocab, default_idf) for tokens in user_tokens]

    sentence_scores = []
    for i, user_vec in enumerate(user_vectors):
        similarities = [calculate_cosine_similarity(user_vec, ideal_vec) for ideal_vec in ideal_vectors]
        best_idx = max(range(len(similarities)), key=lambda j: similarities[j]) if similarities else 0
        best_similarity = max(0.0, similarities[best_idx]) if similarities else 0.0

        sentence_scores.append({
            "sentence": user_sentences[i],
            "best_similarity": round(best_similarity, 4),
            "matched_ideal": ideal_sentences[best_idx] if ideal_sentences else "",
        })

    avg = sum(s["best_similarity"] for s in sentence_scores) / len(sentence_scores) if sentence_scores else 0.0
    return avg, sentence_scores


def cosine_sentence_analysis(ideal_answer: str, user_answer: str) -> dict:
    kw_cov, found, missing = hitung_keyword_coverage(ideal_answer, user_answer)
    avg_sent_cos, sent_detail = hitung_sentence_tfidf_cosine(ideal_answer, user_answer)
    doc_similarity, _tfidf_detail = hitung_tfidf_cosine(ideal_answer, user_answer)

    return {
        "keyword_coverage": kw_cov,
        "keywords_found": found,
        "missing_keywords": missing,
        "sentence_scores": sent_detail,
        "average_sentence_cosine": round(avg_sent_cos, 4),
        "tfidf_cosine_score": round(doc_similarity, 4),  # skala 0-1
    }


def compute_cosine_score(cosine_analysis_result: dict) -> float:
    return round(cosine_analysis_result["tfidf_cosine_score"], 2)