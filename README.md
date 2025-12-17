# 🧠 Cacao Disease Detection ML API (YOLOv8)

This directory contains the machine learning microservice for **AIM-CaD**. It utilizes **YOLOv8 (You Only Look Once)** to perform real-time object detection on cacao pods, identifying diseases such as **Black Pod Rot** and **Vascular Streak Dieback (VSD)**.

This API serves as the inference engine, accepting images via HTTP requests and returning JSON responses with bounding box coordinates and confidence scores.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Model Architecture:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (Nano Version)
* **API Framework:** FastAPI (or Flask)
* **Server:** Uvicorn
* **Computer Vision:** OpenCV (`opencv-python`)

---
