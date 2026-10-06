import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from PIL import Image
import torchvision.transforms as T
from src.models.multimodal_net import MultimodalCropNet
from src.context.engine import ContextEngine
from src.evaluation.gradcam import GradCAM
import json
import cv2
import base64

class InferenceEngine:
    """
    The Multi-Context Inference Engine.

    Coordinates the flow between:
    1. Multimodal ML Model (Predictive)
    2. Runtime Context APIs (Environmental)
    3. Explainability (Grad-CAM)
    """

    def __init__(
        self,
        model_path: str = "models/multimodal_best.pt",
        preprocessor_path: str = "models/tabular_preprocessor.json",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.model = self._load_model(model_path)
        self.preprocessor = self._load_preprocessor(preprocessor_path)
        self.context_engine = ContextEngine()
        self.transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.220, 0.225])
        ])

        # Initialize Grad-CAM on the last conv layer of ResNet18
        # ResNet18 structure: model.image_encoder.layer4
        self.gradcam = GradCAM(self.model, self.model.image_encoder.layer4)

    def _load_model(self, path: str) -> MultimodalCropNet:
        model = MultimodalCropNet()
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def _load_preprocessor(self, path: str) -> Dict[str, Any]:
        with open(path, 'r') as f:
            return json.load(f)

    def preprocess_tabular(self, variety: str, age: int) -> torch.Tensor:
        var_map = self.preprocessor.get("variety_mapping", {})
        var_idx = var_map.get(variety, 0)

        age_mean = self.preprocessor.get("age_mean", 0)
        age_std = self.preprocessor.get("age_std", 1)
        norm_age = (age - age_mean) / (age_std + 1e-6)

        return torch.tensor([[var_idx]], dtype=torch.long).to(self.device), \
               torch.tensor([[norm_age]], dtype=torch.float).to(self.device)

    def _generate_explainability_data(self, img: Image.Image, img_tensor: torch.Tensor, var_tensor: torch.Tensor, age_tensor: torch.Tensor) -> Dict[str, str]:
        """
        Generate Grad-CAM overlay and return as base64 encoded string.
        """
        try:
            # Generate heatmap
            heatmap, _, _ = self.gradcam.generate_heatmap(
                img_tensor,
                variety_idx=var_tensor,
                cont_features=age_tensor
            )

            # Create overlay
            overlay_np = self.gradcam.overlay_heatmap(img, heatmap)

            # Encode to base64
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(overlay_np, cv2.COLOR_RGB2BGR))
            b64_img = base64.b64encode(buffer).decode('utf-8')

            return {
                "status": "available",
                "overlay_b64": f"data:image/jpeg;base64,{b64_img}"
            }
        except Exception as e:
            print(f"Grad-CAM Error: {e}")
            return {"status": "unavailable", "error": str(e)}

    def predict(
        self,
        image_path: str,
        variety: str,
        age: int,
        lat: Optional[float] = None,
        lon: Optional[float] = None
    ) -> Dict[str, Any]:
        # 1. ML Prediction (Core Modalities)
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.transforms(img).unsqueeze(0).to(self.device)
        var_tensor, age_tensor = self.preprocess_tabular(variety, age)

        with torch.no_grad():
            logits = self.model(img_tensor, var_tensor, age_tensor)
            probs = torch.softmax(logits, dim=1)[0]

        top_probs, top_indices = torch.topk(probs, 3)
        class_map = self._get_class_map()

        predictions = []
        for i in range(3):
            idx = top_indices[i].item()
            predictions.append({
                "class": class_map.get(idx, f"Unknown({idx})"),
                "confidence": float(top_probs[i].item())
            })

        # 2. Runtime Context (External APIs)
        location_context = {}
        environment_context = {}
        context_status = {
            "image": "available",
            "variety": "available",
            "age": "available",
            "location": "unavailable",
            "weather": "unavailable"
        }

        if lat is not None and lon is not None:
            full_context = self.context_engine.get_full_context(lat, lon)
            location_context = full_context.get("location", {})
            environment_context = full_context.get("environment", {})

            if location_context.get("status") == "available":
                context_status["location"] = "available"
            if environment_context.get("status") == "available":
                context_status["weather"] = "available"

        # 3. Explainability (Grad-CAM)
        # We need to switch to train mode briefly to get gradients, then back to eval
        self.model.train()
        explainability = self._generate_explainability_data(img, img_tensor, var_tensor, age_tensor)
        self.model.eval()

        return {
            "prediction": predictions[0],
            "top_predictions": predictions,
            "input_context": {
                "variety": variety,
                "age": age
            },
            "location_context": location_context,
            "environment_context": environment_context,
            "context_status": context_status,
            "explainability": explainability
        }

    def _get_class_map(self) -> Dict[int, str]:
        try:
            with open("models/class_mapping.json", 'r') as f:
                mapping = json.load(f)
                return {int(k): v for k, v in mapping.items()}
        except Exception:
            return {}
