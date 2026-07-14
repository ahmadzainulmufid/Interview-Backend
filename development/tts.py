import os
import asyncio
import edge_tts
import glob

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

PROJECT_ROOT = os.path.dirname(BASE_DIR)

AUDIO_FOLDER = os.path.join(PROJECT_ROOT, 'static', 'audio')
os.makedirs(AUDIO_FOLDER, exist_ok=True)

async def save_audio_async(text, filepath):
    communicate = edge_tts.Communicate(
        text,
        "id-ID-GadisNeural"
    )
    await communicate.save(filepath)

def generate_audio_for_question(text, session_id, q_index):
    filename = f"interview_{session_id}_q{q_index}_{os.urandom(4).hex()}.mp3"
    filepath = os.path.join(AUDIO_FOLDER, filename)

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(save_audio_async(text, filepath))
        return f"/static/audio/{filename}"
    except Exception as e:
        print(f"Edge TTS Error: {e}")
        return None
    finally:
        loop.close()

def cleanup_session_audio(session_id):
    try:
        pattern = os.path.join(AUDIO_FOLDER, f"interview_{session_id}_*.mp3")
        audio_files = glob.glob(pattern)

        for file in audio_files:
            try:
                os.remove(file)
            except Exception as e:
                print(f"Gagal menghapus audio {file}: {e}")
    except Exception as e:
        print(f"Cleanup audio error: {e}")

def generate_closing_audio(session_id):
    closing_text = "Terima kasih, wawancara telah selesai. Kami akan memproses wawancara Anda."
    return generate_audio_for_question(closing_text, session_id, "closing")