from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from PIL import Image
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM: Visual Explanations from Deep Networks (Selvaraju et al., 2017).
    
    Extracts gradient-weighted class activation heatmaps from ResNet18 convolutional layer
    to highlight the morphological leaf regions driving disease classification.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        # Register forward and backward hooks
        self.hook_layers()

    def hook_layers(self) -> None:
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        variety_idx: Optional[torch.Tensor] = None,
        cont_features: Optional[torch.Tensor] = None
    ) -> Tuple[np.ndarray, int, float]:
        """
        Compute Grad-CAM heatmap for a single input sample.
        
        Supports both image-only models and multimodal models.
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        if variety_idx is not None and cont_features is not None:
            logits = self.model(input_tensor, variety_idx, cont_features)
        else:
            logits = self.model(input_tensor)

        probs = F.softmax(logits, dim=1)
        pred_class = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0, pred_class].item())

        if target_class is None:
            target_class = pred_class

        # Backward pass targeting class logit
        target_score = logits[0, target_class]
        target_score.backward()

        # Gradients pooled across spatial dimensions: shape (C,)
        pooled_gradients = torch.mean(self.gradients[0], dim=[1, 2])

        # Weight activations by pooled gradients: shape (H, W)
        cam = torch.zeros(self.activations.shape[2:], dtype=torch.float32, device=input_tensor.device)
        for i, w in enumerate(pooled_gradients):
            cam += w * self.activations[0, i, :, :]

        # Apply ReLU to retain only positive influence
        cam = F.relu(cam)
        cam_np = cam.cpu().numpy()

        # Normalize to [0, 1]
        cam_max = np.max(cam_np)
        if cam_max > 1e-7:
            cam_np = cam_np / cam_max
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np, pred_class, confidence

    @staticmethod
    def overlay_heatmap(
        original_image: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """Resize heatmap to image size and overlay with JET colormap."""
        orig_np = np.array(original_image.convert("RGB"))
        h, w = orig_np.shape[:2]

        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        color_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
        color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)

        overlay = np.uint8(alpha * color_heatmap + (1.0 - alpha) * orig_np)
        return overlay

    def save_explanation(
        self,
        original_image: Image.Image,
        input_tensor: torch.Tensor,
        class_names: list,
        output_path: str,
        true_class_name: Optional[str] = None,
        variety_idx: Optional[torch.Tensor] = None,
        cont_features: Optional[torch.Tensor] = None
    ) -> None:
        """Generate and save a 3-panel publication-grade explainability figure."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        heatmap, pred_idx, conf = self.generate_heatmap(
            input_tensor,
            variety_idx=variety_idx,
            cont_features=cont_features
        )
        pred_name = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
        overlay = self.overlay_heatmap(original_image, heatmap)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

        # Panel 1: Original Image
        axes[0].imshow(original_image)
        axes[0].set_title("Input Leaf Image", fontweight="bold", fontsize=11)
        axes[0].axis("off")

        # Panel 2: Grad-CAM Heatmap
        im = axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap (layer4)", fontweight="bold", fontsize=11)
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        # Panel 3: Explainability Overlay
        axes[2].imshow(overlay)
        title_text = f"Prediction: {pred_name} ({conf:.1%})"
        if true_class_name:
            title_text += f"\nTrue Label: {true_class_name}"
        axes[2].set_title(title_text, fontweight="bold", fontsize=11, color="darkgreen" if pred_name == true_class_name else "darkred")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()
