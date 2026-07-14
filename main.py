# main.py
import pandas as pd

from development.auth import app 
from development.cosine_evaluator import build_global_idf
from development.db import db, test_connection
from development.interview_routes import interview_bp
from development.position import position_bp

# Register Blueprint
app.register_blueprint(interview_bp)
app.register_blueprint(position_bp)

if __name__ == '__main__':
    print("🚀 Memulai Aplikasi Backend...")
    
    # 1. Test Koneksi Database
    test_connection(app)

    # 2. Buat Tabel
    with app.app_context():
        # db.drop_all()  
        db.create_all()
        print("✅ [DB] Tabel Database sinkron.")

    df_knowledge = pd.read_csv("data/knowledge_data_indo.csv")
    build_global_idf(df_knowledge["Answer"].dropna().unique().tolist())

    # 3. Jalankan Server
    app.run(host='0.0.0.0', port=5001, debug=True)