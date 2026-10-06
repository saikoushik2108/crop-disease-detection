import io
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Optional, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.utils.config import load_config, get_device
from src.preprocessing.image_transforms import get_eval_transforms
from src.preprocessing.tabular_pipeline import TabularPreprocessor
from src.models.multimodal_net import MultimodalCropNet


app = FastAPI(
    title="Multimodal Crop Disease Detection API",
    description="Inference API for paddy disease classification combining leaf imagery, cultivar variety, and agro-climatic context.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
MODEL_STATE = {
    "model": None,
    "preprocessor": None,
    "config": None,
    "device": None,
    "class_names": []
}


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    num_classes: int
    classes: list


class PredictionResponse(BaseModel):
    predicted_disease: str
    confidence: float
    is_healthy: bool
    top_3_predictions: list
    modality_used: str
    variety: str
    crop_age_days: float
    environmental_context: Dict[str, float]
    advisory_recommendation: str


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


@app.on_event("startup")
def load_artifacts():
    """Load model checkpoint, preprocessor, and configuration upon service startup."""
    try:
        config = load_config("configs/config.yaml")
        device = get_device()
        MODEL_STATE["config"] = config
        MODEL_STATE["device"] = device
        MODEL_STATE["class_names"] = config["dataset"]["classes"]

        preproc_path = Path(config["paths"]["processed_dir"]) / "tabular_preprocessor.json"
        if preproc_path.exists():
            MODEL_STATE["preprocessor"] = TabularPreprocessor.load(str(preproc_path))
            print(f"[API] Tabular Preprocessor loaded from {preproc_path}")
        else:
            print("[API Warning] Tabular Preprocessor not found. Running in fallback mode.")

        # Candidate model checkpoint paths
        candidate_paths = [
            Path("models") / "multimodal_best.pt",
            Path("models/checkpoints") / "multimodal_best.pt",
            Path("models") / "image_only_best.pt",
            Path("models/checkpoints") / "image_only_best.pt",
            Path("models/checkpoints") / "experiment_d_multimodal_fusion_best.pt"
        ]
        ckpt_path = None
        for p in candidate_paths:
            if p.exists():
                ckpt_path = p
                break

        if ckpt_path is not None and MODEL_STATE["preprocessor"] is not None:
            model = MultimodalCropNet(
                num_classes=len(MODEL_STATE["class_names"]),
                num_varieties=MODEL_STATE["preprocessor"].num_varieties,
                num_continuous=MODEL_STATE["preprocessor"].num_continuous_features,
                pretrained=False
            )
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(device)
            model.eval()
            MODEL_STATE["model"] = model
            print(f"[API] Loaded authentic checkpoint from {ckpt_path}")
        else:
            print("[API Warning] No trained checkpoint found. API will return mock diagnostics until trained.")
    except Exception as e:
        print(f"[API Startup Error] {e}")


@app.get("/")
def serve_dashboard():
    """Serve the interactive multimodal web dashboard."""
    frontend_path = PROJECT_ROOT / "frontend" / "index.html"
    if frontend_path.exists():
        return FileResponse(frontend_path)
    return {"message": "Paddy Doctor API Live. Visit /docs for Swagger UI."}


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint confirming API status and model readiness."""
    device_str = str(MODEL_STATE.get("device", "cpu"))
    is_loaded = MODEL_STATE["model"] is not None
    return HealthResponse(
        status="healthy",
        model_loaded=is_loaded,
        device=device_str,
        num_classes=len(MODEL_STATE["class_names"]),
        classes=MODEL_STATE["class_names"]
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_disease(
    image: UploadFile = File(..., description="Crop leaf photograph (JPEG/PNG)"),
    variety: str = Form("ADT37", description="Cultivar / Variety name"),
    age: float = Form(60.0, description="Crop age in days (30 - 90+)"),
    temperature: Optional[float] = Form(28.5, description="Ambient temperature in Celsius"),
    humidity: Optional[float] = Form(78.0, description="Relative humidity percentage"),
    vpd: Optional[float] = Form(0.85, description="Vapor Pressure Deficit in kPa"),
    rain_risk: Optional[float] = Form(0.40, description="Rainfall risk index (0.0 to 1.0)")
):
    """
    Multimodal Inference Endpoint:
    Accepts visual image + agronomic metadata (variety, age) + agro-climatic context,
    returning structured disease diagnosis, confidence, and treatment advisory.
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image (JPEG/PNG)")

    # 1. Process Image
    try:
        image_bytes = await image.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {e}")

    eval_trans = get_eval_transforms(image_size=224)
    img_tensor = eval_trans(pil_img).unsqueeze(0)

    # 2. Process Tabular Context
    preprocessor: TabularPreprocessor = MODEL_STATE["preprocessor"]
    class_names = MODEL_STATE["class_names"]
    device = MODEL_STATE["device"] or torch.device("cpu")

    import pandas as pd
    sample_df = pd.DataFrame([{
        "variety": variety,
        "age": age,
        "temperature": temperature,
        "humidity": humidity,
        "vpd": vpd,
        "rain_risk": rain_risk
    }])

    if preprocessor is not None and preprocessor.is_fitted:
        var_idx_np, cont_feat_np = preprocessor.transform(sample_df)
    else:
        var_idx_np = np.array([0], dtype=np.int64)
        cont_feat_np = np.zeros((1, 5), dtype=np.float32)

    var_tensor = torch.tensor(var_idx_np, dtype=torch.long)
    cont_tensor = torch.tensor(cont_feat_np, dtype=torch.float32)

    # 3. Model Inference
    model = MODEL_STATE["model"]
    if model is not None:
        model.eval()
        with torch.no_grad():
            img_tensor = img_tensor.to(device)
            var_tensor = var_tensor.to(device)
            cont_tensor = cont_tensor.to(device)
            logits = model(img_tensor, var_tensor, cont_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    else:
        # Fallback simulation if running before model training execution
        probs = np.ones(len(class_names)) / len(class_names)

    pred_idx = int(np.argmax(probs))
    pred_disease = class_names[pred_idx]
    confidence = float(probs[pred_idx])

    # Top 3 predictions
    top_indices = np.argsort(probs)[::-1][:3]
    top_3 = [
        {"disease": class_names[i], "confidence": round(float(probs[i]), 4)}
        for i in top_indices
    ]

    is_healthy = (pred_disease == "normal")
    advisory = ADVISORY_DATABASE.get(pred_disease, "Consult your local agricultural extension service.")

    return PredictionResponse(
        predicted_disease=pred_disease,
        confidence=round(confidence, 4),
        is_healthy=is_healthy,
        top_3_predictions=top_3,
        modality_used="Multimodal (Vision + Metadata + Agro-environmental Context)",
        variety=variety,
        crop_age_days=age,
        environmental_context={
            "temperature_C": temperature,
            "humidity_pct": humidity,
            "vpd_kPa": vpd,
            "rain_risk": rain_risk
        },
        advisory_recommendation=advisory
    )
