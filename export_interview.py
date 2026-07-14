import json
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

try:
    from development.auth import app
    from development.db import db, InterviewSession, InterviewQuestion, InterviewAnswer
except ImportError as e:
    raise ImportError(f"Tidak bisa import app atau model database: {e}")

OUTPUT_PATH = "data/hasil_interview_history2.csv"
os.makedirs("data", exist_ok=True)


def export_history(
    only_cosine_based: bool = True,
    min_score:         int  = 0,
    role_filter:       str  = None,
    latest_only:       bool = False,   # ← BARU: ambil 1 sesi terbaru per role
):
    """
    Mengekspor data history wawancara dari database ke CSV.

    Parameters
    ----------
    only_cosine_based : jika True, hanya ekspor jawaban yang memiliki
                        cosine_detail (sesi baru berbasis cosine scoring)
    min_score         : filter minimum skor (default 0 = semua)
    role_filter       : filter berdasarkan role tertentu (opsional)
                        contoh: "Software Engineer"
    latest_only       : jika True, hanya ambil 1 sesi terbaru (berdasarkan
                        completed_at atau created_at). Berguna untuk XAI
                        supaya tidak tercampur data lama.
    """

    with app.app_context():

        query = InterviewSession.query.filter_by(is_completed=True)

        if role_filter:
            query = query.filter(
                InterviewSession.role_focus.ilike(f"%{role_filter}%")
            )

        # Urutkan dari terbaru dulu
        query = query.order_by(InterviewSession.created_at.desc())

        sessions = query.all()
        print(f"\nTotal sesi selesai ditemukan: {len(sessions)}")

        # ── Filter latest_only ───────────────────────────────────────────────
        if latest_only and sessions:
            # Ambil hanya sesi paling baru (index 0 karena sudah desc)
            sessions = [sessions[0]]
            print(f"[latest_only] Hanya memproses sesi terbaru → session_id={sessions[0].id} "
                  f"| role={sessions[0].role_focus} | created={sessions[0].created_at}")

        rows = []
        skipped_no_cosine = 0
        skipped_no_answer = 0

        for session in sessions:

            questions = InterviewQuestion.query.filter_by(
                session_id=session.id
            ).order_by(InterviewQuestion.order_num.asc()).all()

            for q in questions:

                answer = InterviewAnswer.query.filter_by(
                    question_id=q.id
                ).order_by(InterviewAnswer.created_at.desc()).first()

                if not answer:
                    skipped_no_answer += 1
                    continue

                cosine_detail = {}
                has_cosine    = False

                if answer.meta:
                    try:
                        meta_data     = json.loads(answer.meta)
                        cosine_detail = meta_data.get("cosine_detail", {})
                        has_cosine    = bool(cosine_detail)
                    except (json.JSONDecodeError, Exception):
                        cosine_detail = {}
                        has_cosine    = False

                if only_cosine_based and not has_cosine:
                    skipped_no_cosine += 1
                    continue

                score = answer.score or 0
                if score < min_score:
                    continue

                row = {
                    # ── Identitas sesi ──────────────────────────────────────
                    "session_id":         session.id,
                    "user_id":            session.user_id,
                    "role":               session.role_focus,
                    "level":              session.level,
                    "session_created_at": str(session.created_at),

                    # ── Identitas pertanyaan ─────────────────────────────────
                    "question_id":        q.id,
                    "order_num":          q.order_num,
                    "stage":              q.stage,
                    "difficulty_level":   q.difficulty_level,
                    "question":           q.question_text,
                    "ideal_answer":       q.ideal_answer,

                    # ── RAG source ───────────────────────────────────────────
                    "rag_source_id":      q.rag_source_id or "",

                    # ── Jawaban kandidat ─────────────────────────────────────
                    "user_answer":        answer.answer_text,
                    "score":              score,
                    "status":             answer.status or "",
                    "feedback":           answer.feedback or "",

                    # ── Detail cosine (untuk SHAP) ───────────────────────────
                    "keyword_coverage_pct": cosine_detail.get("keyword_coverage_pct", ""),
                    "keywords_found":       ", ".join(cosine_detail.get("keywords_found", [])),
                    "keywords_missing":     ", ".join(cosine_detail.get("keywords_missing", [])),
                    "sentence_cosine":      cosine_detail.get("sentence_cosine", ""),
                    "semantic_coverage":    cosine_detail.get("semantic_coverage", ""),
                    "has_cosine_detail":    has_cosine,

                    # ── Final report LLM (untuk Layer 3 Keyword Attribution) ─
                    "llm_report_conclusion": session.final_report or "",
                }

                rows.append(row)

        if not rows:
            print("\n[WARNING] Tidak ada data yang memenuhi filter untuk diekspor.")
            print(f"  Dilewati karena tidak ada jawaban : {skipped_no_answer}")
            print(f"  Dilewati karena tidak ada cosine  : {skipped_no_cosine}")
            return

        df = pd.DataFrame(rows)
        df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

        # ── Ringkasan ────────────────────────────────────────────────────────
        has_rag    = (df["rag_source_id"] != "").sum()
        has_report = (df["llm_report_conclusion"] != "").sum()

        print(f"\n{'='*55}")
        print(f" Export selesai → {OUTPUT_PATH}")
        print(f"{'='*55}")
        print(f" Total baris diekspor      : {len(df)}")
        print(f" Dilewati (no answer)      : {skipped_no_answer}")
        print(f" Dilewati (no cosine)      : {skipped_no_cosine}")
        print(f" Sesi unik                 : {df['session_id'].nunique()}")
        print(f" Role                      : {df['role'].unique().tolist()}")
        print(f" Stage                     : {df['stage'].value_counts().to_dict()}")
        print(f" Rata-rata skor            : {df['score'].mean():.2f}")
        print(f" Skor tertinggi            : {df['score'].max()}")
        print(f" Skor terendah             : {df['score'].min()}")
        print(f"{'─'*55}")
        print(f" Baris dengan rag_source_id         : {has_rag}/{len(df)}")
        print(f" Baris dengan llm_report_conclusion : {has_report}/{len(df)}")
        print(f"  → Layer 2 & 3 XAI bisa jalan untuk {has_rag} pertanyaan")
        print(f"{'='*55}\n")

        return df


if __name__ == "__main__":
    print("\n=== Export History Wawancara InterviewMate ===\n")

    role = input("Filter role (kosongkan untuk semua): ").strip() or None

    only_cosine = input(
        "Hanya ekspor data berbasis cosine scoring? [Y/n]: "
    ).strip().lower()
    only_cosine = only_cosine != "n"

    latest = input(
        "Hanya ambil sesi wawancara terbaru? [Y/n]: "
    ).strip().lower()
    latest_only = latest != "n"

    if only_cosine:
        print("\n[INFO] Data lama tanpa cosine_detail akan dilewati.")
    else:
        print("\n[INFO] Semua data akan diekspor (termasuk data lama).")

    if latest_only:
        print("[INFO] Hanya mengambil 1 sesi terbaru.")
    else:
        print("[INFO] Semua sesi akan diekspor.")

    export_history(
        only_cosine_based=only_cosine,
        role_filter=role,
        latest_only=latest_only,
    )