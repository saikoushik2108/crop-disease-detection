# Inference API Specification

## 1. Endpoints

### `GET /health`
Returns system status, device hardware in use, and loaded classes.
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu",
  "num_classes": 10,
  "classes": ["bacterial_leaf_blight", "..."]
}
```

### `POST /predict`
Performs multimodal inference on uploaded leaf image and agronomic parameters.
- **Request (Multipart Form):**
  - `image`: Binary file (JPEG/PNG)
  - `variety`: String (e.g., "ADT37", "Ponni")
  - `age`: Float (e.g., 60.0)
  - `temperature`: Float (e.g., 28.5)
  - `humidity`: Float (e.g., 78.0)
  - `vpd`: Float (e.g., 0.85)
  - `rain_risk`: Float (e.g., 0.40)

- **Response:**
```json
{
  "predicted_disease": "blast",
  "confidence": 0.9412,
  "is_healthy": false,
  "top_3_predictions": [
    {"disease": "blast", "confidence": 0.9412},
    {"disease": "brown_spot", "confidence": 0.0381},
    {"disease": "bacterial_leaf_blight", "confidence": 0.0105}
  ],
  "modality_used": "Multimodal (Vision + Metadata + Agro-environmental Context)",
  "variety": "ADT37",
  "crop_age_days": 60.0,
  "environmental_context": {
    "temperature_C": 28.5,
    "humidity_pct": 78.0,
    "vpd_kPa": 0.85,
    "rain_risk": 0.4
  },
  "advisory_recommendation": "Apply systemic fungicides (e.g. tricyclazole). Avoid high nitrogen during cool, humid periods."
}
```
