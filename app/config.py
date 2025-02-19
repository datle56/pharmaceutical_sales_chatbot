# app/config.py
import os
from dotenv import load_dotenv

load_dotenv()  # Load biến môi trường từ file .env nếu có

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set")
