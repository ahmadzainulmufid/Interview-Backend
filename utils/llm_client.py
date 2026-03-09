import os
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Pastikan set environment variable: export GROQ_API_KEY="key_kamu"
groq_api_key = os.environ.get("GROQ_API_KEY", "MASUKKAN_KEY_DI_SINI")
groq_client = Groq(api_key=groq_api_key)