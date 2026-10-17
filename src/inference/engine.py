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
        model_cfg_path = Path("models/model_config.json")
        if model_cfg_path.exists():
            with open(model_cfg_path, 'r') as f:
                cfg = json.load(f)
            model = MultimodalCropNet(
                num_classes=cfg.get("num_classes", 10),
                num_varieties=cfg.get("num_varieties", 11),
                num_continuous=cfg.get("num_continuous", 1),
                variety_embedding_dim=cfg.get("variety_embedding_dim", 16),
                continuous_proj_dim=cfg.get("continuous_feature_dim", 16),
                tabular_embedding_dim=cfg.get("tabular_embedding_dim", 64),
                fusion_hidden_dim=cfg.get("fusion_hidden_dim", 128),
                dropout=cfg.get("dropout", 0.3),
                pretrained=False
            )
        else:
            model = MultimodalCropNet(num_varieties=11, num_continuous=1, pretrained=False)

        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def _load_preprocessor(self, path: str) -> Dict[str, Any]:
        with open(path, 'r') as f:
            return json.load(f)

    def preprocess_tabular(self, variety: str, age: int) -> Tuple[torch.Tensor, torch.Tensor]:
        var_map = self.preprocessor.get("variety_to_idx", self.preprocessor.get("variety_mapping", {}))
        # Clean and alias variety name (including Telangana varieties)
        TELANGANA_VARIETY_ALIASES = {
            "BPT5204": "AndraPonni",
            "BPT 5204": "AndraPonni",
            "SambaMahsuri": "AndraPonni",
            "Samba Mahsuri": "AndraPonni",
            "TelanganaSona": "<unk>",
            "Telangana Sona": "<unk>",
            "RNR15048": "<unk>",
            "RNR 15048": "<unk>",
            "KNM118": "<unk>",
            "KNM 118": "<unk>",
            "MTU1010": "<unk>",
            "MTU 1010": "<unk>",
            "JGL18047": "<unk>",
            "JGL 18047": "<unk>",
            "IR64": "IR20",
            "IR 64": "IR20",
        }
        clean_var = variety.strip()
        mapped_var = TELANGANA_VARIETY_ALIASES.get(clean_var, clean_var)
        var_idx = var_map.get(mapped_var, var_map.get("<unk>", 0))

        scaler = self.preprocessor.get("scaler_params", {}).get("age", {})
        age_mean = scaler.get("mean", self.preprocessor.get("age_mean", 64.02))
        age_std = scaler.get("std", self.preprocessor.get("age_std", 8.97))
        norm_age = (float(age) - age_mean) / (age_std + 1e-6)

        return torch.tensor([var_idx], dtype=torch.long).to(self.device), \
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

        # 2. Runtime Context (Open-Meteo & Nominatim)
        full_context = self.context_engine.get_full_context(lat, lon)
        location_context = full_context.get("location", {})
        environment_context = full_context.get("environment", {})

        context_status = {
            "image": "available",
            "variety": "available",
            "age": "available",
            "location": location_context.get("status", "unavailable"),
            "weather": environment_context.get("status", "unavailable")
        }

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
                result = {}
                for k, v in mapping.items():
                    if isinstance(v, int):
                        result[v] = k
                    else:
                        result[int(k)] = v
                return result
        except Exception:
            return {}
