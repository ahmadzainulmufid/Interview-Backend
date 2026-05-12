# InterviewMate - Backend API 🎙️🤖

Backend untuk sistem simulasi wawancara kerja berbasis suara menggunakan **Speech Recognition**, **Large Language Model (LLM)**, dan **Retrieval-Augmented Generation (RAG)**. Project ini merupakan bagian dari penelitian Tugas Akhir/Skripsi di Politeknik Negeri Jakarta.

## 🚀 Fitur Utama

- **Voice-to-Text & Text-to-Voice**: Pemrosesan input suara menggunakan Groq/Whisper API.
- **RAG System**: Integrasi **ChromaDB** untuk menyimpan dan memanggil data pertanyaan interview yang relevan berdasarkan posisi dan level pekerjaan.
- **AI Interviewer**: Logika wawancara dinamis menggunakan model **LLaMA-3.3-70B**.
- **Authentication**: Keamanan akses menggunakan **JWT (JSON Web Token)**.
- **Database Relasional**: Manajemen data user dan sesi menggunakan **PostgreSQL**.

## 🛠️ Tech Stack

- **Framework**: Flask (Python)
- **Database**: PostgreSQL
- **Vector DB**: ChromaDB
- **LLM API**: Groq Cloud (LLaMA-3.3)
- **Embeddings**: Sentence-Transformers (HuggingFace)
- **Deployment**: Gunicorn, PM2, Nginx, IDCloudHost VPS

## 📋 Prasyarat

Sebelum menjalankan project, pastikan Anda telah menginstal:

- Python 3.11
- PostgreSQL
- FFmpeg (untuk pemrosesan audio)

## ⚙️ Instalasi Lokal

1. **Clone Repositori**

   ```bash
   git clone [https://github.com/ahmadzainulmufid/Interview-Backend.git](https://github.com/ahmadzainulmufid/Interview-Backend.git)
   cd Interview-Backend

   ```

2. **Buat Virtual Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate # Linux/Mac
   # venv\Scripts\activate # Windows

   ```

3. **Install Dependensi**

   ```bash
   pip install -r requirements.txt

   ```

4. **Konfigurasi Environment Variables**

   ```bash
   Buat file .env di root folder:
   DATABASE_URL=postgresql://user:password@localhost:5432/interviewmate_db
   GROQ_API_KEY=gsk_your_api_key_here
   JWT_SECRET_KEY=your_secret_key
   ENV_MODE=development

   ```

5. **Inisialisasi Database**

   ```bash
   python3
   > > > from main import app
   > > > from development.db import db
   > > > with app.app_context():
   > > > ... db.create_all()

   ```

6. **Jalankan Seed Data**

   ```bash
   python3 seed.py

   ```

7. **Jalankan Server**
   ```bash
   python3 main.py
   ```

## Dokumentasi API Endpoints

### Autentikasi

| Method | Endpoint    | Deskripsi                                   | Headers | Request Body (Format)                     |
| :----- | :---------- | :------------------------------------------ | :------ | :---------------------------------------- |
| `POST` | `/register` | Mendaftarkan akun pengguna baru ke database | -       | **JSON**: `username`, `email`, `password` |
| `POST` | `/login`    | Verifikasi kredensial dan mendapatkan JWT   | -       | **JSON**: `email`, `password`             |

### Sesi Wawancara

| Method | Endpoint        | Deskripsi                                                                            | Headers (Wajib)                 | Request Body (Format)                                 |
| :----- | :-------------- | :----------------------------------------------------------------------------------- | :------------------------------ | :---------------------------------------------------- |
| `POST` | `/start`        | Memulai sesi baru, memanggil RAG ChromaDB, & generate pertanyaan pertama             | `Authorization: Bearer <token>` | **JSON**: `role`, `level`                             |
| `POST` | `/answer/audio` | Menerima rekaman suara, transkripsi (Whisper), & generate pertanyaan LLM selanjutnya | `Authorization: Bearer <token>` | **FormData**: `session_id`, `audio` (file .webm/.wav) |
| `POST` | `/end`          | Memaksa sesi berhenti & memicu AI untuk membuat laporan evaluasi akhir               | `Authorization: Bearer <token>` | **JSON**: `session_id`                                |

### Master Data

| Method | Endpoint      | Deskripsi                                                             | Headers (Wajib)                 | Request Body (Format) |
| :----- | :------------ | :-------------------------------------------------------------------- | :------------------------------ | :-------------------- |
| `GET`  | `/categories` | Mengambil daftar kategori, role, dan level untuk pilihan di form awal | `Authorization: Bearer <token>` | -                     |
| `GET`  | `/history`    | Mengambil riwayat sesi wawancara yang pernah dilakukan oleh user      | `Authorization: Bearer <token>` | -                     |
