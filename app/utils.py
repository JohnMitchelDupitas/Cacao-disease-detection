import io, time
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
from .config import settings
from .logger import logger

# thread pool for non-blocking inference
EXECUTOR = ThreadPoolExecutor(max_workers=2)

def load_model(path: str, device: str):
    logger.info(f"Loading model from {path} on device={device}")
    model = YOLO(path)
    # ultralytics handles device selection via model.predict(device=...) at runtime
    return model

def preprocess_image_bytes(file_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return img

def run_inference(model: YOLO, pil_image: Image.Image, imgsz: int = 640, device: str = None):
    """
    Runs inference synchronously (called inside threadpool)
    Returns: (label, confidence, raw_result)
    """
    t0 = time.time()
    # ultralytics supports passing PIL Image directly
    device_arg = device if device else settings.MODEL_DEVICE
    results = model.predict(source=pil_image, imgsz=imgsz, device=device_arg, verbose=False)
    r = results[0]
    # classifier vs detect: for detect -> r.boxes.xyxy, r.boxes.cls, r.boxes.conf
    preds = []
    try:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls.cpu().numpy()[0]) if hasattr(box.cls, "cpu") else int(box.cls)
            conf = float(box.conf.cpu().numpy()[0]) if hasattr(box.conf, "cpu") else float(box.conf)
            preds.append((cls, conf, box.xyxy.cpu().numpy()[0].tolist()))
    except Exception:
        # fallback if no boxes or model type mismatch
        logger.debug("No boxes detected or unexpected result structure")
    elapsed_ms = (time.time() - t0) * 1000.0
    return preds, r, elapsed_ms
