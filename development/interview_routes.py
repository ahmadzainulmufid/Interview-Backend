# development/interview_routes.py
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from development.db import InterviewSession, InterviewQuestion, InterviewAnswer
from development.interview_service import (
    start_interview_session,
    process_candidate_answer,
    transcribe_audio,
    build_session_history,
    force_end_interview
)

interview_bp = Blueprint('interview', __name__)

@interview_bp.route("/start", methods=["POST"])
@jwt_required()
def start_interview():
    current_user_id = get_jwt_identity()

    data = request.get_json() or {}

    role = data.get("role", "Backend Engineer")
    level = data.get("level", "Junior")

    result = start_interview_session(current_user_id, role, level)

    return jsonify({
        "success": True,
        "data": result
    }), 200

@interview_bp.route("/answer/text", methods=["POST"])
def submit_text_answer():

    data = request.get_json() or {}

    session_id = data.get("session_id")
    answer_text = data.get("answer")

    if not session_id or not answer_text:
        return jsonify({
            "success": False,
            "message": "Session ID dan jawaban wajib diisi."
        }), 400

    result = process_candidate_answer(session_id, answer_text)

    if "error" in result:
        return jsonify({
            "success": False,
            "message": result["error"]
        }), 400

    return jsonify({
        "success": True,
        "data": result
    }), 200


@interview_bp.route("/answer/audio", methods=["POST"])
def submit_audio_answer():

    session_id = request.form.get("session_id")
    audio_file = request.files.get("audio")

    if not session_id or not audio_file:
        return jsonify({
            "success": False,
            "message": "Session ID dan file audio wajib diisi."
        }), 400

    try:
        session_id = int(session_id)
    except:
        return jsonify({
            "success": False,
            "message": "Session ID tidak valid."
        }), 400

    transcript = transcribe_audio(audio_file)

    if not transcript:
        return jsonify({
            "success": False,
            "message": "Gagal melakukan transkripsi audio."
        }), 500

    result = process_candidate_answer(session_id, transcript)

    if "error" in result:
        return jsonify({
            "success": False,
            "message": result["error"]
        }), 400

    return jsonify({
        "success": True,
        "transcript": transcript,
        "data": result
    }), 200

@interview_bp.route("/<int:session_id>", methods=["GET"])
@jwt_required()
def get_interview_detail(session_id):
    current_user_id = get_jwt_identity()

    session = InterviewSession.query.filter_by(id=session_id, user_id=current_user_id).first()

    if not session:
        return jsonify({
            "success": False,
            "message": "Session tidak ditemukan."
        }), 404
    
    history = build_session_history(session_id)

    return jsonify({
        "success": True,
        "data": {
            "id": session.id,
            "role": session.role_focus,
            "level": session.level,
            "stage": session.current_stage,
            "difficulty": session.current_difficulty,
            "is_completed": session.is_completed,
            "final_report": session.final_report,
            "created_at": session.created_at,
            "completed_at": session.completed_at,
            "history": history
        }
    }), 200

@interview_bp.route("/history", methods=["GET"])
@jwt_required()
def get_all_history():
    try:
        current_user_id = get_jwt_identity()

        sessions = InterviewSession.query.filter_by(user_id=current_user_id)\
                   .order_by(InterviewSession.created_at.desc()).all()
        
        result = []
        for s in sessions:
            date_str = s.created_at.strftime("%d %b %Y") if s.created_at else "Unknown Date"

            answers = InterviewAnswer.query.join(InterviewQuestion).filter(InterviewQuestion.session_id == s.id).all()
            valid_scores = [a.score for a in answers if a.score is not None]
            avg_score = round(sum(valid_scores) / len(valid_scores)) if valid_scores else 0
            
            result.append({
                "id": s.id,
                "role": s.role_focus,
                "level": s.level,
                "date": date_str,
                "score": avg_score,
                "is_completed": s.is_completed
            })
            
        return jsonify({
            "success": True,
            "data": result
        }), 200
        
    except Exception as e:
        print("Error fetching history:", e)
        return jsonify({
            "success": False,
            "message": "Gagal mengambil data history."
        }), 500

@interview_bp.route("/end", methods=["POST"])
def end_interview_early():
    data = request.get_json() or {}
    session_id = data.get("session_id")

    if not session_id:
        return jsonify({"success": False, "message": "Session ID wajib diisi."}), 400

    result = force_end_interview(session_id)

    if "error" in result:
        return jsonify({"success": False, "message": result["error"]}), 400

    return jsonify({"success": True, "message": result["message"]}), 200