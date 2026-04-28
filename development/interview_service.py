from multiprocessing import util
import os
import io
import asyncio
import edge_tts
from pydub import AudioSegment
from dotenv import load_dotenv
import pandas as pd
import chromadb
import json
from chromadb.utils import embedding_functions
from groq import Groq
from sqlalchemy import text
from sympy import content
from sentence_transformers import util, SentenceTransformer
from gtts import gTTS
from development.db import db
from development.db import InterviewSession, InterviewQuestion, InterviewAnswer

# Load environment variables
load_dotenv()

ENV_MODE = os.getenv("ENV_MODE", "development")

# Setup Global
chroma_client = None
collection = None

# embedder = SentenceTransformer("all-MiniLM-L6-v2")
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Konfigurasi Path 
BASE_DIR = os.path.abspath(os.path.dirname(__file__)) 
PROJECT_ROOT = os.path.dirname(BASE_DIR) 
AUDIO_FOLDER = os.path.join(PROJECT_ROOT, 'static', 'audio')
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'knowledge_data_rag_adaptive_final.csv')

os.makedirs(AUDIO_FOLDER, exist_ok=True)

def init_knowledge_base():
    global chroma_client, collection

    if ENV_MODE == "production":
        count = db.session.execute(text("SELECT count(*) FROM knowledge_base")).scalar()
        if count > 0:
            return True # Sudah terisi

        if not os.path.exists(DATA_PATH): return False

        print("[PROD] Memasukkan CSV ke Supabase Vector...")
        df = pd.read_csv(DATA_PATH).fillna("General Concept")

        for index, row in df.iterrows():
            doc_text = row["Embedding_Text"]
            emb = embedder.encode(doc_text).tolist() 
            meta = {
                "role": str(row["Role"]), "stage": str(row["Stage"]),
                "difficulty_level": int(row["Adaptive_Level"]), "answer": str(row["Answer"])
            }
            rag_id = f"q_{index}"

            insert_query = text("""
                INSERT INTO knowledge_base (id, content, metadata, embedding)
                VALUES (:id, :content, :metadata, :embedding)
            """)
            db.session.execute(insert_query, {
                "id": rag_id, "content": doc_text,
                "metadata": json.dumps(meta), "embedding": str(emb)
            })
            
        db.session.commit()
        print("[PROD] Selesai migrasi CSV ke Supabase!")
        return True

    else:
        if collection: return collection

        chroma_path = os.path.join(PROJECT_ROOT, 'chroma_storage')
        chroma_client = chromadb.PersistentClient(path=chroma_path)
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        collection = chroma_client.get_or_create_collection(name="adaptive_interview_rag", embedding_function=embed_fn)

        if collection.count() == 0:
            if not os.path.exists(DATA_PATH): return None

            print("[DEV] Memasukkan CSV ke ChromaDB Lokal...")
            df = pd.read_csv(DATA_PATH).fillna("General Concept")
            documents = df["Embedding_Text"].tolist()
            ids = [f"q_{i}" for i in range(len(df))]
            metadatas = []
            for _, row in df.iterrows():
                metadatas.append({
                    "role": str(row["Role"]), "stage": str(row["Stage"]),
                    "difficulty_level": int(row["Adaptive_Level"]), "answer": str(row["Answer"])
                })
            collection.add(documents=documents, metadatas=metadatas, ids=ids)
            print("[DEV] Selesai migrasi CSV ke ChromaDB!")
            
        return collection

def get_groq_client():
    # Konsisten pakai TTS_API_KEY
    api_key = os.getenv("TTS_API_KEY") 
    if not api_key:
        raise ValueError("ERROR: TTS_API_KEY belum diset di file .env")
    return Groq(api_key=api_key)

def generate_audio_for_question(text, session_id, q_index):
    """Menghasilkan file audio mp3 dari teks pertanyaan menggunakan Edge TTS"""
    filename = f"interview_{session_id}_q{q_index}_{os.urandom(4).hex()}.mp3"
    filepath = os.path.join(AUDIO_FOLDER, filename)
    try:
        # Menggunakan suara Gadis (Wanita Indonesia)
        communicate = edge_tts.Communicate(text, "id-ID-GadisNeural")
        asyncio.run(communicate.save(filepath))
        return f"/static/audio/{filename}"
    except Exception as e:
        print(f"Edge TTS Error: {e}")
        return None

def transcribe_audio(audio_file):
    """Convert audio to text using Groq Whisper."""
    client = get_groq_client()
    try:
        file_content = audio_file.read()
        filename = audio_file.filename if audio_file.filename else "audio.wav"
        file_tuple = (filename, file_content)

        transcription = client.audio.transcriptions.create(
            file=file_tuple,
            model="whisper-large-v3",
            response_format="json",  
            language="id", 
            temperature=0.0          
        )
        return transcription.text
    except Exception as e:
        print(f"Error Transcribing Audio: {e}")
        return None
    
def extract_experience_topics(user_answer):
    client = get_groq_client()
    prompt = f"""Dari jawaban perkenalan kandidat berikut, ekstrak 3-5 kata kunci berupa teknologi utama atau arsitektur spesifik.
    Jawaban: {user_answer}
    Output format JSON: {{"topics": []}}"""
    try:
        res = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}], temperature=0.2)
        content = res.choices[0].message.content.replace("```json","").replace("```","").strip()
        return json.loads(content).get("topics", [])
    except:
        return []

def evaluate_answer_ai(question, ideal_answer, user_answer):
    ideal_emb = embedder.encode(ideal_answer)
    user_emb = embedder.encode(user_answer)

    cosine_raw = util.cos_sim(ideal_emb, user_emb).item()
    cosine_score = max(0, round(cosine_raw * 100))

    client = get_groq_client()
    prompt = f"""Bertindak sebagai penilai interview.
    Pertanyaan: {question}
    Kunci Referensi: {ideal_answer}
    Jawaban Kandidat: {user_answer}
    Output format JSON: {{"score": 0-100, "feedback": "Kritik singkat 2 kalimat", "status": "Good/Average/Bad"}}"""
    try:
        res = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}], temperature=0.2)
        content = res.choices[0].message.content.replace("```json","").replace("```","").strip()
        llm_result = json.loads(content)
        
        llm_score = int(llm_result.get("score", 50)) 
        feedback = llm_result.get("feedback", "Jawaban cukup baik.")
        
    except Exception as e: 
        print(f"[DEBUG] Error parsing JSON dari LLM: {e}")
        llm_score = 50
        feedback = "Evaluasi LLM error, diasumsikan standar."

    final_score = round((llm_score * 0.7) + (cosine_score * 0.3))

    if final_score > 75:
        status = "Good"
    elif final_score > 50:
        status = "Average"
    else:
        status = "Bad"

    print(f"[DEBUG EVALUASI] LLM Score: {llm_score} | Cosine Score: {cosine_score} | Final: {final_score}")

    return {
        "score": final_score,
        "feedback": feedback,
        "status": status
    }

def generate_interview_question(context_text, ideal_answer, role, difficulty_label, background=None, stage="Technical", previous_questions=None):
    client = get_groq_client()
    
    # Menyesuaikan konteks pertanyaan dengan tech stack kandidat jika ada
    bg_str = ", ".join(background) if background else ""
    bg_instruction = ""
    if bg_str:
        bg_instruction = f"""
        - SESUAIKAN KONTEKS: Kandidat memiliki pengalaman dengan teknologi ini: {bg_str}.
        - JIKA memungkinkan, jadikan salah satu teknologi tersebut sebagai konteks atau contoh kasus di dalam pertanyaan Anda.
        """

    # Mencegah Redundansi
    prev_q_prompt = ""
    if previous_questions:
        prev_q_prompt = f"\nPENTING: JANGAN ulangi atau gunakan topik yang mirip dengan daftar pertanyaan sebelumnya berikut ini:\n- " + "\n- ".join(previous_questions)

    # Menyesuaikan instruksi berdasarkan stage wawancara
    if stage == "Study Case":
        tugas = "Berikan sebuah studi kasus (Study Case) atau skenario problem-solving berskala arsitektur/sistem."
    elif stage == "Soft Skill":
        tugas = "Berikan pertanyaan wawancara tentang Soft Skill (komunikasi, penyelesaian konflik, manajemen waktu)."
    else:
        tugas = "Buat 1 pertanyaan wawancara teknis spesifik."

    # Prompt Utama
    prompt = f"""Anda adalah Senior Tech Interviewer berbahasa Indonesia untuk posisi {role} (Level: {difficulty_label}).
    
    Topik Inti (Bahan Pertanyaan): {context_text}
    Ekspektasi Jawaban: {ideal_answer}
    {prev_q_prompt}
    
    Instruksi:
    1. Tugas Anda: {tugas}
    {bg_instruction}
    2. Fokus pada 'Topik Inti'.
    3. Langsung berikan pertanyaan tanpa kata pengantar, sapaan, atau basa-basi.
    """
    
    try:
        # Gunakan temperature yang sedikit lebih tinggi (0.7) agar kalimat lebih variatif
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile", 
            messages=[{"role": "user", "content": prompt}], 
            temperature=0.7 
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        print("Error Generating Question:", e)
        return f"Tolong jelaskan tentang: {context_text}"

def retrieve_adaptive_question(role, stage, difficulty, used_ids):
    init_knowledge_base()

    if ENV_MODE == "production":
        query_text = f"{role} {stage} level {difficulty}"
        query_embedding = embedder.encode(query_text).tolist()

        if stage == "Technical":
            sql = text("""
                SELECT id, content, metadata FROM knowledge_base
                WHERE metadata->>'role' = :role AND metadata->>'stage' = :stage AND (metadata->>'difficulty_level')::int = :diff
                ORDER BY embedding <=> :emb::vector LIMIT 20
            """)
            params = {"role": role, "stage": stage, "diff": difficulty, "emb": str(query_embedding)}
        else:
            sql = text("""
                SELECT id, content, metadata FROM knowledge_base
                WHERE metadata->>'role' = :role AND metadata->>'stage' = :stage
                ORDER BY embedding <=> :emb::vector LIMIT 20
            """)
            params = {"role": role, "stage": stage, "emb": str(query_embedding)}

        results = db.session.execute(sql, params).fetchall()
        for row in results:
            if row.id not in used_ids:
                return row.content, row.metadata, row.id
        return None, None, None

    else:
        global collection
        where_filter = {"$and": [{"role": role}, {"stage": stage}]}
        if stage == "Technical":
            where_filter["$and"].append({"difficulty_level": difficulty})

        results = collection.query(
            query_texts=[f"{role} {stage} level {difficulty}"],
            n_results=20,
            where=where_filter
        )

        if not results["documents"] or not results["documents"][0]:
            return None, None, None

        for i, doc in enumerate(results["documents"][0]):
            rag_id = results["ids"][0][i]
            if rag_id not in used_ids:
                return doc, results["metadatas"][0][i], rag_id
        return None, None, None

def generate_final_report_ai(history, role):
    client = get_groq_client()
    comp_block = "\n".join([f"Q: {h['question']} | A: {h['user_answer']} | Score: {h['score']}" for h in history])
    prompt = f"""Buat laporan Gap Analysis wawancara untuk posisi {role} berdasarkan data berikut:
    {comp_block}
    Berikan summary, kekuatan teknis, kelemahan, dan rekomendasi belajar."""
    res = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}], temperature=0.2)
    return res.choices[0].message.content

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
            history.append({
                "stage": q.stage,
                "question": q.question_text,
                "user_answer": answer.answer_text,
                "score": answer.score,
                "feedback": answer.feedback
            })

    return history

def start_interview_session(user_id, role="Backend Engineer", level="Junior"):
    """
    Membuat session baru dan menyimpan opening question ke DB
    """

    # 1. Buat Session
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

    # 2. Buat Opening Question
    opening_question = (
        f"Halo! Silakan perkenalkan diri Anda dan ceritakan pengalaman Anda "
        f"sebagai {role}, termasuk teknologi utama yang paling sering Anda gunakan."
    )

    audio_url = generate_audio_for_question(opening_question, session_id, 1)

    new_question = InterviewQuestion(
        session_id=session_id,
        question_text=opening_question,
        ideal_answer="Menjelaskan background dan teknologi utama.",
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

def process_candidate_answer(session_id, user_answer):

    try:
        session = InterviewSession.query.get(session_id)

        if not session or session.is_completed:
            return {"error": "Session tidak valid."}

        last_question = InterviewQuestion.query.filter_by(
            session_id=session_id
        ).order_by(InterviewQuestion.order_num.desc()).first()

        if not last_question:
            return {"error": "Pertanyaan tidak ditemukan."}

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
            status=eval_result["status"]
        )

        db.session.add(answer)

        # Adaptive Difficulty
        if session.current_stage == "Technical":
            if eval_result["score"] > 75:
                session.current_difficulty = min(3, session.current_difficulty + 1)
            elif eval_result["score"] < 50:
                session.current_difficulty = max(1, session.current_difficulty - 1)

        tech_count = InterviewQuestion.query.filter_by(
            session_id=session_id,
            stage="Technical"
        ).count()

        if session.current_stage == "Opening":
            session.current_stage = "Technical"

        elif session.current_stage == "Technical" and tech_count >= 5:
            session.current_stage = "Study Case"

        elif session.current_stage == "Study Case":
            history = build_session_history(session_id)
            final_report = generate_final_report_ai(history, session.role_focus)

            session.final_report = final_report
            session.is_completed = True

            db.session.commit()

            return {
                "stage": "Completed",
                "final_report": final_report
            }

        # Anti duplicate RAG
        used_ids = [
            q.rag_source_id
            for q in InterviewQuestion.query.filter_by(session_id=session_id).all()
            if q.rag_source_id
        ]

        doc, meta, rag_id = retrieve_adaptive_question(
            f"{session.level} {session.role_focus}",
            session.current_stage,
            session.current_difficulty,
            used_ids
        )

        ideal = meta["answer"] if meta else "Jawaban sesuai best practice."
        context = doc if doc else "Konsep umum."

        diff_label = ["Easy", "Medium", "Hard"][session.current_difficulty - 1]

        past_questions_records = InterviewQuestion.query.filter_by(session_id=session_id).order_by(InterviewQuestion.order_num.asc()).all()
        recent_questions = [q.question_text for q in past_questions_records[-4:]] if past_questions_records else []

        next_question = generate_interview_question(
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
            ideal_answer=ideal,
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
    
def force_end_interview(session_id):
    try:
        session = InterviewSession.query.get(session_id)
        if not session:
            return {"error": "Session tidak valid."}
        
        if session.is_completed:
            return {"message": "Wawancara sudah selesai."}

        # Kumpulkan semua history yang sudah ada
        history = build_session_history(session_id)
        
        # Minta AI buat final report berdasarkan obrolan yang terpotong ini
        if history:
            final_report = generate_final_report_ai(history, session.role_focus)
        else:
            final_report = "Kandidat mengakhiri wawancara sebelum memberikan jawaban apa pun."

        session.final_report = final_report
        session.is_completed = True
        db.session.commit()

        return {"success": True, "message": "Wawancara diakhiri secara paksa dan laporan disimpan."}
    
    except Exception as e:
        db.session.rollback()
        print("FORCE END ERROR:", e)
        return {"error": "Gagal mengakhiri wawancara secara paksa."}