import os
import re
from groq import Groq
from dotenv import load_dotenv
from development.normalization import FILLERS, TECH_TERM_MAP

load_dotenv()

def get_stt_client():
    api_key = os.getenv("TTS_API_KEY") 
    if not api_key:
        raise ValueError("ERROR: TTS_API_KEY (Groq) belum diset di file .env")
    return Groq(api_key=api_key)

def normalize_transcript(text):
    result = text

    for wrong, correct in TECH_TERM_MAP.items():
        result = re.sub(rf"\b{re.escape(wrong)}\b", correct, result, flags=re.I)

    for filler in FILLERS:
        result = re.sub(rf"\b{filler}\b", "", result, flags=re.I)

    result = re.sub(r"\s+", " ", result)
    
    return result.strip()

def filter_hallucinated_segments(segments):
    clean_texts = []
    for seg in segments:
        no_speech = seg.get("no_speech_prob", 0.0)
        avg_logprob = seg.get("avg_logprob", 0.0)
 
        if no_speech > 0.6 and avg_logprob < -1.0:
            print(f"[STT] Segmen dibuang (kemungkinan halusinasi): '{seg.get('text', '').strip()}'")
            continue
 
        clean_texts.append(seg.get("text", ""))
 
    return " ".join(clean_texts).strip()

def transcribe_audio(audio_file):
    client = get_stt_client()
    try:
        audio_file.seek(0)
        file_content = audio_file.read()
        filename = audio_file.filename if audio_file.filename else "audio.wav"
        file_tuple = (filename, file_content)

        transcription = client.audio.transcriptions.create(
            file=file_tuple,
            model="whisper-large-v3",
            response_format="verbose_json",  
            language="id", 
            temperature=0.0,
            prompt="""
            Wawancara kerja bidang teknologi informasi dalam Bahasa Indonesia.
            Transkripsikan seluruh ucapan dalam Bahasa Indonesia sesuai yang diucapkan.
            Pertahankan istilah teknis dalam bentuk aslinya, seperti REST API, React.js,
            JavaScript, Docker, Kubernetes, PostgreSQL, MySQL, CI/CD, DevOps,
            Golang, Python, FIX protocol, InfluxDB, dan microservices.
            Jangan menerjemahkan kalimat Bahasa Indonesia menjadi Bahasa Inggris.
            """      
        )
        raw_text = filter_hallucinated_segments(transcription.segments)
        
        normalized = normalize_transcript(raw_text)

        return normalized
    
    except Exception as e:
        print(f"Error Transcribing Audio: {e}")
        return None