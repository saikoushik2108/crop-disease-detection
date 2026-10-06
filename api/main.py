from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import os
from typing import Optional
from pydantic import BaseModel
from src.inference.engine import InferenceEngine

app = FastAPI(
    title="Paddy Disease AI API",
    description="Multimodal Multi-Context Aware Deep Learning System for Paddy Disease Classification",
    version="1.0.0"
)

# CORS Configuration for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Inference Engine Instance
# Initialized once on startup to avoid reloading model on every request
engine = InferenceEngine()

class PredictionResponse(BaseModel):
    prediction: dict
    top_predictions: list
    input_context: dict
    location_context: dict
    environment_context: dict
    context_status: dict

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
async def predict_disease(
    image: UploadFile = File(...),
    variety: str = Form(...),
    age: int = Form(...),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None)
):
    try:
        # 1. Save uploaded image temporarily
        temp_path = f"temp_{image.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # 2. Run Inference
        result = engine.predict(
            image_path=temp_path,
            variety=variety,
            age=age,
            lat=lat,
            lon=lon
        )

        # 3. Cleanup temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
