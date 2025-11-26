from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from .config import settings
from .logger import logger
from .schemas import PredictResponse, Prediction, HealthCheck
from .utils import preprocess_image_bytes, load_model, run_inference, EXECUTOR
import asyncio
import time

app = FastAPI(title="FarmIQ Cacao ML API")

# CORS
if settings.ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# global model object
MODEL = None
MODEL_PATH = settings.MODEL_PATH

@app.on_event("startup")
async def startup_event():
    global MODEL
    try:
        MODEL = load_model(MODEL_PATH, settings.MODEL_DEVICE)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.exception("Failed to load model during startup: %s", e)
        MODEL = None

def verify_api_key(x_api_key: Optional[str] = Header(None)):
    if not settings.API_KEY:
        return True
    if x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

@app.get("/health", response_model=HealthCheck)
async def health():
    return HealthCheck(status="ok", model_loaded=(MODEL is not None), model_path=MODEL_PATH)

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...), x_api_key: str = Depends(verify_api_key)):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if file.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=400, detail="File must be an image")

    raw = await file.read()
    pil = preprocess_image_bytes(raw)

    loop = asyncio.get_event_loop()
    preds, raw_result, elapsed_ms = await loop.run_in_executor(EXECUTOR, run_inference, MODEL, pil, 640, settings.MODEL_DEVICE)

    # format predictions: map class id -> name
    results = []
    for cls, conf, box in preds:
        name = settings.MODEL_CLASS_NAMES[cls] if settings.MODEL_CLASS_NAMES and len(settings.MODEL_CLASS_NAMES) > cls else str(cls)
        results.append(Prediction(disease=name, confidence=conf, model_version=MODEL_PATH, processing_time_ms=elapsed_ms))

    # If no boxes, return empty prediction list but still successful
    if not results:
        # optional: return a default 'no detection' with low confidence
        results = [Prediction(disease="No detection", confidence=0.0, model_version=MODEL_PATH, processing_time_ms=elapsed_ms)]

    # log
    logger.info("Predicted %d objects, time=%.1fms", len(results), elapsed_ms)
    return PredictResponse(predictions=results)

@app.post("/predict/batch", response_model=PredictResponse)
async def predict_batch(files: List[UploadFile] = File(...), x_api_key: str = Depends(verify_api_key)):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    loop = asyncio.get_event_loop()
    all_predictions = []
    for file in files:
        if file.content_type.split("/")[0] != "image":
            continue
        raw = await file.read()
        pil = preprocess_image_bytes(raw)
        preds, raw_result, elapsed_ms = await loop.run_in_executor(EXECUTOR, run_inference, MODEL, pil, 640, settings.MODEL_DEVICE)
        if not preds:
            all_predictions.append(Prediction(disease="No detection", confidence=0.0, model_version=MODEL_PATH, processing_time_ms=elapsed_ms))
            continue
        # take highest confidence for this image (or include all)
        best = max(preds, key=lambda x: x[1])
        name = settings.MODEL_CLASS_NAMES[best[0]] if settings.MODEL_CLASS_NAMES and len(settings.MODEL_CLASS_NAMES) > best[0] else str(best[0])
        all_predictions.append(Prediction(disease=name, confidence=best[1], model_version=MODEL_PATH, processing_time_ms=elapsed_ms))

    logger.info("Batch predicted %d images", len(all_predictions))
    return PredictResponse(predictions=all_predictions)
