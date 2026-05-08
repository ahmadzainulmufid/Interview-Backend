# auth.py
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_jwt_extended import (
    JWTManager, jwt_required, get_jwt_identity, 
    create_access_token, create_refresh_token, get_jwt
)
from flask_cors import CORS  # <--- IMPORT CORS
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta, timezone

# Import dari file db.py yang berada di satu folder
from development.db import db, User, TokenBlocklist

load_dotenv()

base_dir = os.path.abspath(os.path.dirname(__file__)) # Folder development/
root_dir = os.path.dirname(base_dir) # Folder interview-backend/
static_folder_path = os.path.join(root_dir, 'static')

app = Flask(__name__, static_folder=static_folder_path, static_url_path='/static')

ENV_MODE = os.getenv("ENV_MODE", "development")

# --- KONFIGURASI DATABASE ---
if ENV_MODE == "development":
    # 1. Mode Lokal Laptop 
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:admin@localhost:5432/interview_db'

elif ENV_MODE == "supabase":
    # 2. Mode Supabase 
    app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("SUPABASE_DATABASE_URL")

elif ENV_MODE == "production":
    # 3. Mode VPS IDCloudHost 
    app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL")

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# --- KONFIGURASI JWT ---
app.config['JWT_SECRET_KEY'] = os.getenv("JWT_SECRET_KEY")
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(minutes=15)
app.config['JWT_REFRESH_TOKEN_EXPIRES'] = timedelta(days=30)

# --- INIT EXTENSIONS ---
db.init_app(app)
jwt = JWTManager(app)

# --- SETUP CORS ---
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}}, supports_credentials=True)

# Callback Blocklist
@jwt.token_in_blocklist_loader
def check_if_token_revoked(jwt_header, jwt_payload):
    jti = jwt_payload["jti"]
    token = db.session.query(TokenBlocklist.id).filter_by(jti=jti).scalar()
    return token is not None

# --- ROUTES ---
@app.route('/register', methods=['POST'])
def register():
    username = request.json.get('username')
    email = request.json.get('email')
    password = request.json.get('password')
    confirm_password = request.json.get('confirm_password')

    if not username or not email or not password or not confirm_password:
        return jsonify({"msg": "Semua kolom wajib diisi"}), 400

    if password != confirm_password:
        return jsonify({"msg": "Password tidak cocok"}), 400

    existing_user = User.query.filter((User.username == username) | (User.email == email)).first()

    if existing_user:
        return jsonify({"msg": "Username atau Email sudah terdaftar"}), 409

    hashed_password = generate_password_hash(password)
    new_user = User(username=username, email=email, password=hashed_password)
    
    db.session.add(new_user)
    db.session.commit()

    return jsonify({"msg": "Registrasi berhasil"}), 201

@app.route('/login', methods=['POST'])
def login():
    username = request.json.get('username')
    password = request.json.get('password')

    if not username or not password:
        return jsonify({"msg": "Data tidak lengkap"}), 400

    user = User.query.filter_by(username=username).first()

    if not user or not check_password_hash(user.password, password):
        return jsonify({"msg": "Username atau Password salah"}), 401

    access_token = create_access_token(identity=str(user.id))
    refresh_token = create_refresh_token(identity=str(user.id))
    
    return jsonify({
        "msg": "Login berhasil",
        "access_token": access_token, 
        "refresh_token": refresh_token,
        "user_details": {
            "id": user.id,
            "username": user.username,
            "email": user.email
        }
    }), 200

@app.route('/refresh', methods=['POST'])
@jwt_required(refresh=True)
def refresh():
    current_user = get_jwt_identity()
    new_access_token = create_access_token(identity=current_user)
    return jsonify(access_token=new_access_token), 200

@app.route('/logout', methods=['DELETE'])
@jwt_required()
def logout():
    jti = get_jwt()["jti"]
    now = datetime.now(timezone.utc)
    db.session.add(TokenBlocklist(jti=jti, created_at=now))
    db.session.commit()
    return jsonify({"msg": "Berhasil logout"}), 200

@app.route('/protected', methods=['GET'])
@jwt_required()
def protected():
    current_user_id = get_jwt_identity()
    user = db.session.get(User, current_user_id)
    return jsonify(
        logged_in_as_id=current_user_id,
        username=user.username
    ), 200