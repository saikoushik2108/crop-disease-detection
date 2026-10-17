from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel
from src.inference.engine import InferenceEngine
from src.context.advisory import build_detailed_advisory_report

app = FastAPI(
    title="Paddy Disease AI API",
    description="Multimodal Multi-Context Aware Deep Learning System for Paddy Disease Classification",
    version="1.0.0"
)

# CORS Configuration for React Frontend & Local Development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Global Inference Engine Instance
engine = InferenceEngine()

ADVISORY_DATABASE = {
    "bacterial_leaf_blight": "Apply copper hydroxide or valid bactericide. Avoid excessive nitrogen application. Drain field temporarily.",
    "bacterial_leaf_streak": "Maintain balanced fertilizer application. Disinfect agricultural tools. Avoid working in fields when foliage is wet.",
    "bacterial_panicle_blight": "Ensure clean certified seed. Avoid excessive late nitrogen top-dressing.",
    "blast": "Apply systemic fungicides (e.g. tricyclazole or isoprothiolane). Avoid high nitrogen during cool, humid periods.",
    "brown_spot": "Correct soil potassium and zinc deficiencies. Apply balanced NPK fertilizers and seed treatment fungicides.",
    "dead_heart": "Monitor for yellow stem borer moths. Release Trichogramma parasitoids or apply recommended biological agents.",
    "downy_mildew": "Improve field drainage. Avoid stagnant water in seedbeds. Remove infected alternate grass hosts.",
    "hispa": "Clip leaf tips before transplanting to destroy eggs. Spray neem seed kernel extract or recommended bio-pesticides.",
    "normal": "Crop foliage shows healthy vigorous growth. Continue standard integrated crop management practices.",
    "tungro": "Control green leafhopper vectors. Rogue and bury infected plants immediately. Plant resistant cultivars."
}

class PredictionResponse(BaseModel):
    prediction: dict
    top_predictions: list
    input_context: dict
    location_context: dict
    environment_context: dict
    context_status: dict
    explainability: Optional[dict] = None
    # Compatibility fields for frontend dashboards
    predicted_disease: Optional[str] = None
    confidence: Optional[float] = None
    is_healthy: Optional[bool] = None
    top_3_predictions: Optional[list] = None
    advisory_recommendation: Optional[str] = None
    modality_used: Optional[str] = None
    variety: Optional[str] = None
    crop_age_days: Optional[float] = None
    detailed_report: Optional[dict] = None

@app.get("/")
def serve_dashboard():
    """Serve the interactive multimodal web dashboard."""
    frontend_path = PROJECT_ROOT / "frontend" / "index.html"
    if frontend_path.exists():
        return FileResponse(frontend_path)
    return {"message": "Paddy Disease AI API Live. Visit /docs for Swagger UI."}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(engine.device),
        "num_classes": 10
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_disease(
    image: UploadFile = File(...),
    variety: str = Form("ADT45"),
    age: float = Form(65.0),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None)
):
    temp_path = f"temp_{image.filename}"
    try:
        # 1. Save uploaded image temporarily
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # 2. Run Inference
        int_age = int(round(age))
        result = engine.predict(
            image_path=temp_path,
            variety=variety,
            age=int_age,
            lat=lat,
            lon=lon
        )

        pred_disease = result["prediction"]["class"]
        pred_conf = result["prediction"]["confidence"]

        # Populate compatibility fields
        result["predicted_disease"] = pred_disease
        result["confidence"] = round(pred_conf, 4)
        result["is_healthy"] = (pred_disease == "normal")
        result["top_3_predictions"] = [
            {"disease": p["class"], "confidence": round(p["confidence"], 4)}
            for p in result.get("top_predictions", [])
        ]
        result["advisory_recommendation"] = ADVISORY_DATABASE.get(
            pred_disease, "Consult your local agricultural extension service."
        )
        result["modality_used"] = "Multimodal (Vision + Cultivar + Plant Age)"
        # 3. Comprehensive Epidemiological & Advisory Report
        detailed_report = build_detailed_advisory_report(
            disease_key=pred_disease,
            confidence=pred_conf,
            variety=variety,
            age=int_age,
            weather_data=result.get("environment_context", {}),
            location_data=result.get("location_context", {})
        )
        result["detailed_report"] = detailed_report

        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 3. Cleanup temporary file safely
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
