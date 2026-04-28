import pandas as pd
from deep_translator import GoogleTranslator
import time

def translate_csv(input_file, output_file):
    # Load data
    df = pd.read_csv(input_file)
    translator = GoogleTranslator(source='en', target='id')

    print(f"Memulai terjemahan untuk {len(df)} baris...")

    # Fungsi untuk menerjemahkan dengan penanganan error
    def safe_translate(text):
        try:
            if pd.isna(text): return ""
            return translator.translate(str(text))
        except Exception as e:
            print(f"Error: {e}")
            return text

    # Terjemahkan kolom Pertanyaan dan Jawaban (Sesuaikan nama kolom dengan CSV-mu)
    # Misal nama kolomnya 'question' dan 'ideal_answer'
    df['Question'] = df['Question'].apply(safe_translate)
    df['Answer'] = df['Answer'].apply(safe_translate)

    # Simpan hasil
    df.to_csv(output_file, index=False)
    print(f"Selesai! File tersimpan di: {output_file}")

# Jalankan
translate_csv('data/knowledge_data_rag_adaptive_final.csv', 'data/knowledge_data_indo.csv')