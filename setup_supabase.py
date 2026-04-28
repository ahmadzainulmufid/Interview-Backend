import os
from flask import Flask
from dotenv import load_dotenv
from development.db import db

load_dotenv()

app = Flask(__name__)
supabase_database_url = os.getenv("SUPABASE_DATABASE_URL")

if not supabase_database_url:
   print("❌ ERROR: SUPABASE_DATABASE_URL tidak ditemukan di .env!")
   exit()

app.config['SQLALCHEMY_DATABASE_URI'] = supabase_database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

with app.app_context():
    print("Memproses pembuatan tabel ke Supabase...")
    db.create_all()
    print("✅ BERHASIL! Semua tabel sudah dibuat di Supabase.")