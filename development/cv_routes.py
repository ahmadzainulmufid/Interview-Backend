# development/cv_routes.py
from flask import Blueprint, request, jsonify
from development.db import db, Candidate
from development.utils import extract_text_from_pdf
from development.interview_service import analyze_profile

cv_bp = Blueprint('cv', __name__)

@cv_bp.route('/upload-cv', methods=['POST'])
def upload_cv():
    if 'file' not in request.files:
        return jsonify({"msg": "No file part"}), 400
    
    file = request.files['file']
    name = request.form.get('name', 'Unknown Candidate')

    if file and file.filename != '':
        # 1. Ekstrak Teks
        extracted_text = extract_text_from_pdf(file)
        if not extracted_text:
            return jsonify({"msg": "Gagal membaca PDF"}), 400

        # 2. Simpan ke DB (Buat row baru)
        new_candidate = Candidate(
            name=name,
            filename=file.filename,
            cv_text=extracted_text
        )
        db.session.add(new_candidate)
        db.session.commit()

        # 3. REAL AI ANALYSIS (Panggil fungsi dari interview_service.py)
        # Fungsi ini akan return dict: {role_hint, skills, match_score}
        ai_result = analyze_profile(extracted_text)

        # Mapping hasil AI ke format JSON yang diminta Frontend
        # Kita berikan default value jika AI gagal mendeteksi
        response_data = {
            "role": ai_result.get("role_hint", "General Applicant"),
            "matchScore": ai_result.get("match_score", 75),
            # Ambil 5 skill pertama
            "hardSkills": ai_result.get("skills", [])[:5], 
            # Mockup soft skills (karena biasanya tidak diekstrak detail oleh script simple)
            "softSkills": ["Communication", "Adaptability", "Problem Solving"] 
        }

        return jsonify({
            "msg": "Success",
            "candidate_id": new_candidate.id,
            "analysis": response_data
        }), 201
    
    return jsonify({"msg": "Upload gagal"}), 400

@cv_bp.route('/delete-cv/<int:candidate_id>', methods=['DELETE'])
def delete_cv(candidate_id):
    candidate = Candidate.query.get(candidate_id)
    
    if not candidate:
        return jsonify({"msg": "Data not found"}), 404
    
    try:
        # Hapus data dari database
        db.session.delete(candidate)
        db.session.commit()
        return jsonify({"msg": "CV data deleted successfully"}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"msg": f"Error deleting data: {str(e)}"}), 500