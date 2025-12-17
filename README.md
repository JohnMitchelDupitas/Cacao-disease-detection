# Cacao ML API (production-ready FastAPI)

## Requirements
- Python 3.9+ (3.11 recommended)
- GPU recommended for inference speed
- Docker (optional)

## Setup (local)
1. Copy `.env.example` to `.env` and set API_KEY, MODEL_PATH, MODEL_DEVICE, ALLOWED_ORIGINS.
2. Install deps:
   ```bash
   py -m pip install -r requirements.txt
