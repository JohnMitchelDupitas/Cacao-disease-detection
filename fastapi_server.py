from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import uvicorn

app = FastAPI(title="Cacao Disease Detection API")

# 1. Allow the Mobile App to talk to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load the Model
# Ensure 'runs/detect/train/weights/best.pt' exists!
model = YOLO("runs/detect/train/weights/best.pt")

# 3. NAME TRANSLATOR (Scientific -> Common English)
# This maps the ID from your JSON to a readable name
CLASS_NAMES = {
    0: "Healthy",
    1: "Pod Borer (Carmenta)",           # carmenta foraseminis
    2: "Witches Broom",                  # moniliophthora perniciosa
    3: "Frosty Pod Rot",                 # moniliophthora roreri
    4: "Black Pod Disease"               # phytophthora palmivora
}

@app.get("/")
def home():
    return {"message": "Cacao API is running with 5 classes!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read Image
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Predict (Confidence Threshold = 70%)
    results = model.predict(img, conf=0.7)
    result = results[0]

    detections = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        
        # Get the Friendly Name (or fallback to scientific if missing)
        friendly_name = CLASS_NAMES.get(class_id, result.names[class_id])
        
        # Get Coordinates
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "class": friendly_name,
            "scientific_name": result.names[class_id], # Keep scientific name too just in case
            "confidence": round(confidence, 2),
            "box": {
                "x1": round(x1), "y1": round(y1),
                "x2": round(x2), "y2": round(y2)
            }
        })

    return {
        "filename": file.filename,
        "detections": detections,
        "message": "Success" if detections else "No disease detected"
    }

if __name__ == "__main__":
    # Running on Port 8001 to avoid conflict with Laravel (8000)
    uvicorn.run(app, host="0.0.0.0", port=8001)