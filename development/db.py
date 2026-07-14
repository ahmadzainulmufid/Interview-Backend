# db.py
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
import enum

# Inisialisasi object SQLAlchemy
db = SQLAlchemy()

class LevelEnum(enum.Enum):
    JUNIOR = "Junior"
    INTERMEDIATE = "Intermediate"
    SENIOR = "Senior"

# --- DEFINISI MODELS ---
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    sessions = db.relationship('InterviewSession', backref='user', lazy=True)

    def __repr__(self):
        return f'<User {self.username}>'

class TokenBlocklist(db.Model):
    __tablename__ = 'token_blocklist'
    id = db.Column(db.Integer, primary_key=True)
    jti = db.Column(db.String(36), nullable=False, index=True)
    created_at = db.Column(db.DateTime, nullable=False)

class InterviewSession(db.Model):
    __tablename__ = 'interview_sessions'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    # Informasi Interview
    role_focus = db.Column(db.String(100), nullable=False)   
    level = db.Column(db.String(50), nullable=False)     

    # Adaptive State
    current_stage = db.Column(db.String(50), default="Opening")
    current_difficulty = db.Column(db.Integer, default=1)

    # Status Interview
    is_completed = db.Column(db.Boolean, default=False)

    # Final Report AI
    final_report = db.Column(db.Text, nullable=True)
    final_score = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=db.func.now())
    completed_at = db.Column(db.DateTime, nullable=True)

    # Relationship
    questions = db.relationship(
        'InterviewQuestion',
        backref='session',
        lazy=True,
        cascade="all, delete-orphan"
    )

class InterviewQuestion(db.Model):
    __tablename__ = 'interview_questions'
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(
        db.Integer,
        db.ForeignKey('interview_sessions.id'),
        nullable=False
    )
    question_text = db.Column(db.Text, nullable=False)
    ideal_answer = db.Column(db.Text, nullable=True)
    stage = db.Column(db.String(50))  
    difficulty_level = db.Column(db.Integer)  
    audio_path = db.Column(db.String(255))
    order_num = db.Column(db.Integer)
    rag_source_id = db.Column(db.String(100), nullable=True)
    created_at = db.Column(db.DateTime, default=db.func.now())
    answers = db.relationship(
        'InterviewAnswer',
        backref='question',
        lazy=True,
        cascade="all, delete-orphan"
    )

class InterviewAnswer(db.Model):
    __tablename__ = 'interview_answers'
    id = db.Column(db.Integer, primary_key=True)
    question_id = db.Column(
        db.Integer,
        db.ForeignKey('interview_questions.id'),
        nullable=False
    )
    answer_text = db.Column(db.Text, nullable=True)
    recording_url = db.Column(db.String(255), nullable=True)
    score = db.Column(db.Integer)  
    feedback = db.Column(db.Text)
    status = db.Column(db.String(20)) 
    meta = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=db.func.now())
    
class Category(db.Model):
    __tablename__ = "categories"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)
    created_at = db.Column(db.DateTime, default=db.func.now())
    roles = db.relationship("Role", backref="category", lazy=True)

class Role(db.Model):
    __tablename__ = "roles"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    category_id = db.Column(db.Integer, db.ForeignKey("categories.id"), nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.now())
    positions = db.relationship("Position", backref="role", lazy=True)

class Position(db.Model):
    __tablename__ = "positions"
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    role_id = db.Column(db.Integer, db.ForeignKey("roles.id"), nullable=False)
    level = db.Column(db.Enum(LevelEnum), nullable=False)
    description = db.Column(db.Text, nullable=True)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=db.func.now())
    updated_at = db.Column(
        db.DateTime,
        default=db.func.now(),
        onupdate=db.func.now()
    )

# Fungsi Test Koneksi (Opsional)
def test_connection(app_instance):
    try:
        with app_instance.app_context():
            db.session.execute(text('SELECT 1'))
            print("\n[DB] Koneksi Database Berhasil!\n")
    except Exception as e:
        print(f"\n[DB] Gagal Konek: {e}\n")