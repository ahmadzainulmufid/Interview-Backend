import os
import json
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from development.db import db
from development.db import InterviewSession, InterviewQuestion, InterviewAnswer
from development.tts import generate_audio_for_question, cleanup_session_audio, generate_closing_audio
from development.answer_evaluator import evaluate_answer_ai
from development.final_report import generate_final_report_ai

load_dotenv()

ENV_MODE = os.getenv("ENV_MODE", "development")

chroma_client = None
collection = None

# embedder = SentenceTransformer("all-MiniLM-L6-v2")
# embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

BASE_DIR = os.path.abspath(os.path.dirname(__file__)) 
PROJECT_ROOT = os.path.dirname(BASE_DIR) 
AUDIO_FOLDER = os.path.join(PROJECT_ROOT, 'static', 'audio')
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'knowledge_data_indo.csv')

os.makedirs(AUDIO_FOLDER, exist_ok=True)

OPENING_IDEAL_ANSWERS = {
    "Software Engineer": (
        "Saya seorang Software Engineer yang berpengalaman merancang, membangun, dan memelihara "
        "sistem perangkat lunak, mulai dari struktur data dan algoritma hingga desain sistem dan API. "
        "Teknologi yang paling sering saya gunakan meliputi bahasa pemrograman berorientasi objek, "
        "database relasional, serta praktik seperti REST API dan version control."
    ),
    "Data Scientist": (
        "Saya seorang Data Scientist yang berpengalaman mengolah data mentah menjadi wawasan yang "
        "dapat ditindaklanjuti. Teknologi yang paling sering saya gunakan meliputi Python dengan "
        "library seperti pandas dan scikit-learn, deep learning framework, serta SQL."
    ),
    "Frontend Developer": (
        "Saya seorang Frontend Developer yang berpengalaman membangun antarmuka pengguna yang "
        "responsif dan mudah digunakan. Teknologi yang paling sering saya gunakan meliputi React, "
        "Vue, atau Angular, dipadukan dengan HTML, CSS, dan JavaScript modern."
    ),
    "Backend Developer": (
        "Saya seorang Backend Developer yang berpengalaman merancang dan membangun server, API, "
        "serta sistem database yang skalabel dan aman."
    ),
    "DevOps Engineer": (
        "Saya seorang DevOps Engineer yang berpengalaman mengotomasi proses build, testing, dan "
        "deployment. Teknologi yang paling sering saya gunakan meliputi pipeline CI/CD, cloud "
        "platform, serta Docker dan Kubernetes."
    ),
    "QA Engineer": (
        "Saya seorang QA Engineer yang berpengalaman memastikan kualitas software melalui pengujian "
        "manual maupun otomasi."
    ),
    "Product Manager": (
        "Saya seorang Product Manager yang berpengalaman menyelaraskan kebutuhan pengguna, bisnis, "
        "dan tim engineering. Saya biasa menggunakan Agile/Scrum dan tools seperti Jira atau Notion."
    ),
    "Ux Designer": (
        "Saya seorang UX Designer yang berpengalaman merancang pengalaman pengguna berdasarkan riset. "
        "Tools yang paling sering saya gunakan meliputi Figma, Sketch, atau Adobe XD."
    ),
}

DEFAULT_OPENING_IDEAL = (
    "Menjelaskan latar belakang profesional, pengalaman relevan, dan teknologi/tools utama "
    "yang paling sering digunakan sesuai dengan role yang dilamar."
)

DEFAULT_TECHNICAL_IDEAL = (
    "Menjelaskan konsep secara teknis dengan contoh penerapan nyata sesuai standar industri "
    "untuk role ini."
)

def init_knowledge_base():

    global chroma_client, collection

    if collection:
        return collection

    chroma_path = os.path.join(
        PROJECT_ROOT,
        'chroma_storage'
    )

    chroma_client = chromadb.PersistentClient(
        path=chroma_path
    )

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

    collection = chroma_client.get_or_create_collection(
        name="adaptive_interview_rag",
        embedding_function=embed_fn
    )

    if collection.count() == 0:

        if not os.path.exists(DATA_PATH):
            return None

        print("Memproses CSV ke ChromaDB...")

        df = pd.read_csv(DATA_PATH).fillna(
            "General Concept"
        )

        documents = df["Embedding_Text"].tolist()

        ids = [
            f"q_{i}"
            for i in range(len(df))
        ]

        metadatas = []

        for _, row in df.iterrows():

            metadatas.append({
                "role": str(row["Role"]),
                "stage": str(row["Stage"]),
                "difficulty_level": int(row["Adaptive_Level"]),
                "answer": str(row["Answer"])
            })

        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        print("Proses Menyimpan CSV ke ChromaDB!")

    return collection

def get_groq_client():
    api_key = os.getenv("TTS_API_KEY") 
    if not api_key:
        raise ValueError("ERROR: TTS_API_KEY belum diset di file .env")
    return Groq(api_key=api_key)
    
def extract_experience_topics(user_answer):
    client = get_groq_client()
    system_instruction = f"""Dari jawaban perkenalan kandidat berikut, ekstrak 3-5 kata kunci berupa teknologi utama atau arsitektur spesifik.
    Jawaban: {user_answer}
    Output format JSON: {{"topics": []}}"""
    try:
        res = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": system_instruction}], temperature=0.2)
        content = res.choices[0].message.content.replace("```json","").replace("```","").strip()
        return json.loads(content).get("topics", [])
    except Exception as e:
        print(f"Error extracting experience topics: {e}")
        return []

def generate_interview_question(context_text, ideal_answer, role, difficulty_label, background=None, stage="Technical", previous_questions=None):
    client = get_groq_client()
    
    bg_str = ", ".join(background) if background else ""
    bg_instruction = ""
    if bg_str:
        bg_instruction = f"""
        - SESUAIKAN KONTEKS: Kandidat memiliki pengalaman dengan teknologi ini: {bg_str}.
        - JIKA memungkinkan, jadikan salah satu teknologi tersebut sebagai konteks atau contoh kasus di dalam pertanyaan Anda.
        """

    previous_questions_instruction = ""
    if previous_questions:
        previous_questions_instruction = f"\nPENTING: JANGAN ulangi atau gunakan topik yang mirip dengan daftar pertanyaan sebelumnya berikut ini:\n- " + "\n- ".join(previous_questions)

    if stage == "Case Study":
        tugas = "Berikan sebuah studi kasus (Case Study) atau skenario problem-solving berskala arsitektur/sistem."
    elif stage == "Soft Skill":
        tugas = "Berikan pertanyaan wawancara tentang Soft Skill (komunikasi, penyelesaian konflik, manajemen waktu)."
    else:
        tugas = "Buat 1 pertanyaan wawancara teknis spesifik."

    system_instruction = f"""Anda adalah Senior Tech Interviewer berbahasa Indonesia untuk posisi {role} (Level: {difficulty_label}).
    
    Topik Inti (Bahan Pertanyaan): {context_text}
    Ekspektasi Jawaban: {ideal_answer}
    {previous_questions_instruction}
    
    Instruksi:
    1. Tugas Anda: {tugas}
    {bg_instruction}
    2. Fokus pada 'Topik Inti'.
    3. Buat juga 'ideal_answer' baru yang SPESIFIK menjawab pertanyaan yang Anda buat 
    (boleh dikembangkan dari Referensi Jawaban Ideal, tapi harus benar-benar cocok 
    dengan pertanyaan final Anda).
    4. Output HARUS JSON murni, tanpa markdown/backtick, format:
    {{"question": "...", "ideal_answer": "..."}}
    """
    
    try:
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile", 
            messages=[{"role": "user", "content": system_instruction}], 
            temperature=0.7 
        )
        content = res.choices[0].message.content.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(content)
        return parsed.get("question", "").strip(), parsed.get("ideal_answer", ideal_answer).strip()
    except Exception as e:
        print("Error Generating Question:", e)
        return f"Tolong jelaskan tentang: {context_text}", ideal_answer

def retrieve_adaptive_question(role, stage, difficulty, used_ids):
    global collection
    init_knowledge_base()

    def try_query(role_filter):
        where_filter = {
            "$and": [
                {"role": role_filter},
                {"stage": stage}
            ]
        }
        if stage == "Technical":
            where_filter["$and"].append({"difficulty_level": difficulty})

        return collection.query(
            query_texts=[f"{role_filter} {stage} level {difficulty}"],
            n_results=20,
            where=where_filter
        )

    results = try_query(role)

    if not results["documents"] or not results["documents"][0]:
        role_clean = role
        for prefix in ["Junior ", "Intermediate ", "Senior "]:
            if role.startswith(prefix):
                role_clean = role[len(prefix):]
                break

        if role_clean != role:
            print(f"[RAG] Fallback role query: '{role}' → '{role_clean}'")
            results = try_query(role_clean)

    if not results["documents"] or not results["documents"][0]:
        return None, None, None

    for i, doc in enumerate(results["documents"][0]):
        rag_id = results["ids"][0][i]
        if rag_id not in used_ids:
            return doc, results["metadatas"][0][i], rag_id

    return None, None, None

def build_session_history(session_id):

    questions = InterviewQuestion.query.filter_by(
        session_id=session_id
    ).order_by(InterviewQuestion.order_num.asc()).all()

    history = []

    for q in questions:
        answer = InterviewAnswer.query.filter_by(
            question_id=q.id
        ).order_by(InterviewAnswer.created_at.desc()).first()

        if answer:
            cosine_detail = {}
            if hasattr(answer, 'meta') and answer.meta:
                try:
                    cosine_detail = json.loads(answer.meta).get("cosine_detail", {})
                except Exception:
                    pass

            history.append({
                "stage": q.stage,
                "question": q.question_text,
                "user_answer": answer.answer_text,
                "score": answer.score,
                "feedback": answer.feedback,
                "cosine_detail": cosine_detail
            })

    return history

def start_interview_session(user_id, role="Backend Engineer", level="Junior"):
    """
    Membuat session baru dan menyimpan opening question ke DB
    """

    new_session = InterviewSession(
        user_id=user_id,
        role_focus=role,
        level=level,
        current_stage="Opening",
        current_difficulty=1,
        is_completed=False
    )

    db.session.add(new_session)
    db.session.commit()

    session_id = new_session.id

    opening_question = (
        f"Halo! Silakan perkenalkan diri Anda dan ceritakan pengalaman Anda "
        f"sebagai {role}, termasuk teknologi utama yang paling sering Anda gunakan."
    )

    audio_url = generate_audio_for_question(opening_question, session_id, 1)

    new_question = InterviewQuestion(
        session_id=session_id,
        question_text=opening_question,
        ideal_answer=OPENING_IDEAL_ANSWERS.get(role, DEFAULT_OPENING_IDEAL),
        stage="Opening",
        difficulty_level=1,
        audio_path=audio_url,
        order_num=1
    )

    db.session.add(new_question)
    db.session.commit()

    return {
        "session_id": session_id,
        "stage": "Opening",
        "question": opening_question,
        "audio_url": audio_url
    }

def process_candidate_answer(current_user_id, session_id, user_answer):

    try:
        session = InterviewSession.query.filter_by(
            id=session_id,
            user_id=current_user_id
        ).first()

        if not session or session.is_completed:
            return {"error": "Session tidak valid."}
        
        max_duration = timedelta(minutes=11)

        if datetime.utcnow() - session.created_at > max_duration:

            session.is_completed = True
            session.completed_at = datetime.utcnow()

            db.session.commit()

            return {
            "error": "Waktu wawancara telah habis."
            }

        last_question = InterviewQuestion.query.filter_by(
            session_id=session_id
        ).order_by(InterviewQuestion.order_num.desc()).first()

        if not last_question:
            return {"error": "Pertanyaan tidak ditemukan."}
        
        existing_answer = InterviewAnswer.query.filter_by(
            question_id=last_question.id
        ).first()

        if existing_answer:
            return {
            "error": "Pertanyaan ini sudah dijawab."
        }

        eval_result = evaluate_answer_ai(
            last_question.question_text,
            last_question.ideal_answer,
            user_answer
        )

        answer = InterviewAnswer(
            question_id=last_question.id,
            answer_text=user_answer,
            score=eval_result["score"],
            feedback=eval_result["feedback"],
            status=eval_result["status"],
            meta=json.dumps({"cosine_detail": eval_result.get("cosine_detail", {})})
        )

        db.session.add(answer)

        if session.current_stage == "Technical":
            if eval_result["status"] == "Good":
                session.current_difficulty = min(3, session.current_difficulty + 1)
            elif eval_result["status"] == "Bad":
                session.current_difficulty = max(1, session.current_difficulty - 1)

        tech_count = InterviewQuestion.query.filter_by(
            session_id=session_id,
            stage="Technical"
        ).count()

        if session.current_stage == "Opening":
            session.current_stage = "Technical"

        elif session.current_stage == "Technical" and tech_count >= 5:
            session.current_stage = "Case Study"

        elif session.current_stage == "Case Study":
            history = build_session_history(session_id)
            final_report = generate_final_report_ai(history, session.role_focus)

            session.final_report = final_report
            session.is_completed = True
            session.completed_at = datetime.utcnow()

            db.session.commit()

            cleanup_session_audio(session_id)

            closing_audio = generate_closing_audio(session_id)

            return {
                "stage": "Completed",
                "final_report": final_report,
                "closing_audio": closing_audio
            }

        used_ids = [
            q.rag_source_id
            for q in InterviewQuestion.query.filter_by(session_id=session_id).all()
            if q.rag_source_id
        ]

        doc, meta, rag_id = retrieve_adaptive_question(
            session.role_focus,
            session.current_stage,
            session.current_difficulty,
            used_ids
        )
        diff_label = ["Easy", "Medium", "Hard"][session.current_difficulty - 1]

        ideal = meta["answer"] if meta else DEFAULT_TECHNICAL_IDEAL
        context = doc if doc else (
            f"Topik teknis umum yang relevan untuk posisi {session.role_focus} "
            f"level {diff_label if 'diff_label' in dir() else session.current_difficulty}."
        )
        
        past_questions_records = InterviewQuestion.query.filter_by(session_id=session_id).order_by(InterviewQuestion.order_num.asc()).all()
        recent_questions = [q.question_text for q in past_questions_records[-4:]] if past_questions_records else []

        next_question, next_ideal_answer = generate_interview_question(
            context_text=context,
            ideal_answer=ideal,
            role=f"{session.level} {session.role_focus}",
            difficulty_label=diff_label,
            stage=session.current_stage,
            previous_questions=recent_questions
        )

        order = InterviewQuestion.query.filter_by(
            session_id=session_id
        ).count() + 1

        audio_url = generate_audio_for_question(next_question, session_id, order)

        new_q = InterviewQuestion(
            session_id=session_id,
            question_text=next_question,
            ideal_answer=next_ideal_answer,
            stage=session.current_stage,
            difficulty_level=session.current_difficulty,
            order_num=order,
            audio_path=audio_url,
            rag_source_id=rag_id
        )

        db.session.add(new_q)
        db.session.commit()

        return {
            "stage": session.current_stage,
            "evaluation_previous_answer": eval_result,
            "next_question": next_question,
            "audio_url": audio_url
        }

    except Exception as e:
        db.session.rollback()
        print("PROCESS ERROR:", e)
        return {"error": "Terjadi kesalahan sistem."}
    
def force_end_interview(current_user_id, session_id):
    try:
        session = InterviewSession.query.filter_by(
            id=session_id,
            user_id=current_user_id
        ).first()

        if not session:
            return {"error": "Session tidak valid."}
        
        if session.is_completed:
            return {"message": "Wawancara sudah selesai."}

        history = build_session_history(session_id)
        
        if history:
            final_report = generate_final_report_ai(history, session.role_focus)
        else:
            final_report = "Kandidat mengakhiri wawancara sebelum memberikan jawaban apa pun."

        session.final_report = final_report
        session.is_completed = True
        session.completed_at = datetime.utcnow()
        
        db.session.commit()

        cleanup_session_audio(session_id)

        return {"success": True, "message": "Wawancara diakhiri secara paksa dan laporan disimpan."}
    
    except Exception as e:
        db.session.rollback()
        print("FORCE END ERROR:", e)
        return {"error": "Gagal mengakhiri wawancara secara paksa."}
        