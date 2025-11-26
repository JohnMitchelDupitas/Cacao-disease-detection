import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    MODEL_PATH = os.getenv("MODEL_PATH", "runs/detect/train/weights/best.pt")
    MODEL_DEVICE = os.getenv("MODEL_DEVICE", "cpu")
    MODEL_NUM_CLASSES = int(os.getenv("MODEL_NUM_CLASSES", "5"))
    MODEL_CLASS_NAMES = [s.strip() for s in os.getenv("MODEL_CLASS_NAMES", "").split(",")] if os.getenv("MODEL_CLASS_NAMES") else []
    API_KEY = os.getenv("API_KEY", "change_me")
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8001"))
    WORKERS = int(os.getenv("WORKERS", "1"))
    ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
    LOG_FILE = os.getenv("LOG_FILE", None)

settings = Settings()
