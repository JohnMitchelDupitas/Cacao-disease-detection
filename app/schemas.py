from pydantic import BaseModel
from typing import List, Optional

class Prediction(BaseModel):
    disease: str
    confidence: float
    model_version: Optional[str]
    processing_time_ms: Optional[float]

class PredictResponse(BaseModel):
    predictions: List[Prediction]

class HealthCheck(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
