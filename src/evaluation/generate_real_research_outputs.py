import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch

from src.utils.config import load_config
from src.preprocessing.image_transforms import get_eval_transforms
from src.preprocessing.tabular_pipeline import TabularPreprocessor
from src.models.baseline_image import ImageBaselineResNet
from src.models.multimodal_net import MultimodalCropNet
from src.evaluation.gradcam import GradCAM


def main():
    print("[Generating Real Ablation Research Artifacts]")
    results_dir = Path("results")
    metrics_dir = results_dir / "metrics"
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Metrics
    with open(metrics_dir / "image_only_test_metrics.json") as f:
        img_metrics = json.load(f)
    with open(metrics_dir / "metadata_only_test_metrics.json") as f:
        meta_metrics = json.load(f)
    with open(metrics_dir / "multimodal_test_metrics.json") as f:
        multi_metrics = json.load(f)

    # 2. Build Summary Comparison Table
    summary_data = [
        {
            "Experiment": "Exp B: Metadata-Only (Context Baseline)",
            "Modality": "Variety + Plant Age (MLP)",
            "Accuracy (%)": round(meta_metrics["accuracy"] * 100, 2),
            "Macro F1": round(meta_metrics["macro_f1"], 4),
            "Macro Precision": round(meta_metrics["macro_precision"], 4),
            "Macro Recall": round(meta_metrics["macro_recall"], 4),
            "Weighted F1": round(meta_metrics["weighted_f1"], 4),
        },
        {
            "Experiment": "Exp A: Image-Only (Visual Baseline)",
            "Modality": "RGB Leaf Image (ResNet18)",
            "Accuracy (%)": round(img_metrics["accuracy"] * 100, 2),
            "Macro F1": round(img_metrics["macro_f1"], 4),
            "Macro Precision": round(img_metrics["macro_precision"], 4),
            "Macro Recall": round(img_metrics["macro_recall"], 4),
            "Weighted F1": round(img_metrics["weighted_f1"], 4),
        },
        {
            "Experiment": "Exp C: Multimodal Fusion (Proposed)",
            "Modality": "Visual + Variety + Age",
            "Accuracy (%)": round(multi_metrics["accuracy"] * 100, 2),
            "Macro F1": round(multi_metrics["macro_f1"], 4),
            "Macro Precision": round(multi_metrics["macro_precision"], 4),
            "Macro Recall": round(multi_metrics["macro_recall"], 4),
            "Weighted F1": round(multi_metrics["weighted_f1"], 4),
        }
    ]

    summary_df = pd.DataFrame(summary_data)
    csv_path = metrics_dir / "real_ablation_comparison.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"  --> Saved comparison summary to: {csv_path}")

    # 3. Class-by-Class Comparative Table
    classes = list(img_metrics["per_class"].keys())
    per_class_rows = []
    for cls in classes:
        per_class_rows.append({
            "Class Name": cls,
            "Support": img_metrics["per_class"][cls]["support"],
            "Metadata F1": round(meta_metrics["per_class"][cls]["f1"], 4),
            "Image F1": round(img_metrics["per_class"][cls]["f1"], 4),
            "Multimodal F1": round(multi_metrics["per_class"][cls]["f1"], 4),
            "Delta Multi vs Image": round(multi_metrics["per_class"][cls]["f1"] - img_metrics["per_class"][cls]["f1"], 4)
        })
    per_class_df = pd.DataFrame(per_class_rows)
    per_class_csv = metrics_dir / "real_per_class_ablation.csv"
    per_class_df.to_csv(per_class_csv, index=False)
    print(f"  --> Saved per-class comparison to: {per_class_csv}")

    # 4. Generate Comparative Bar Chart
    plt.figure(figsize=(14, 7), dpi=200)
    x = np.arange(len(classes))
    width = 0.26

    f1_meta = [meta_metrics["per_class"][c]["f1"] for c in classes]
    f1_img = [img_metrics["per_class"][c]["f1"] for c in classes]
    f1_multi = [multi_metrics["per_class"][c]["f1"] for c in classes]

    plt.bar(x - width, f1_meta, width, label="Metadata-Only (MLP)", color="#94a3b8", alpha=0.9)
    plt.bar(x, f1_img, width, label="Image-Only (ResNet18)", color="#3b82f6", alpha=0.9)
    plt.bar(x + width, f1_multi, width, label="Multimodal (Image + Context)", color="#10b981", alpha=0.9)

    plt.xlabel("Disease Category", fontsize=11, fontweight="bold")
    plt.ylabel("Test Macro F1-Score", fontsize=11, fontweight="bold")
    plt.title("Ablation Study: Per-Class F1 Score on Authentic Paddy Doctor Test Set (1,562 samples)", fontsize=13, fontweight="bold", pad=12)
    plt.xticks(x, [c.replace("_", "\n") for c in classes], fontsize=9)
    plt.ylim(0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0", fontsize=10)
    plt.tight_layout()

    fig_chart_path = figures_dir / "real_per_class_f1_comparison.png"
    plt.savefig(fig_chart_path)
    plt.close()
    print(f"  --> Saved comparative figure to: {fig_chart_path}")

    # 5. Generate Grad-CAM Visual Interpretability on Real Test Images
    print("  --> Generating Grad-CAM on authentic test images...")
    config = load_config("configs/config.yaml")
    class_names = config["dataset"]["classes"]
    label_to_idx = {name: i for i, name in enumerate(class_names)}
    idx_to_label = {i: name for name, i in label_to_idx.items()}

    # Load trained ImageBaseline model
    device = torch.device("cpu")
    model_img = ImageBaselineResNet(num_classes=len(class_names), pretrained=False)
    ckpt = torch.load("models/image_only_best.pt", map_location=device, weights_only=False)
    model_img.load_state_dict(ckpt["model_state_dict"])
    model_img.eval()

    gradcam = GradCAM(model=model_img, target_layer=model_img.layer4)
    eval_transforms = get_eval_transforms(224)

    # Select representative test images from test.csv
    test_df = pd.read_csv("data/processed/splits/test.csv")
    sample_classes = ["blast", "dead_heart", "bacterial_panicle_blight", "normal"]
    samples_to_plot = []

    for c in sample_classes:
        match = test_df[test_df["label"] == c].iloc[0]
        samples_to_plot.append((match["image_id"], match["label"], match["variety"], match["age"]))

    fig, axes = plt.subplots(len(samples_to_plot), 3, figsize=(11, 3.2 * len(samples_to_plot)), dpi=200)

    for i, (img_id, true_label, variety, age) in enumerate(samples_to_plot):
        img_path = Path("data/raw/train_images") / true_label / img_id
        if not img_path.exists():
            continue
        orig_img = Image.open(img_path).convert("RGB")
        tensor_img = eval_transforms(orig_img).unsqueeze(0)

        true_idx = label_to_idx[true_label]
        heatmap, pred_idx, conf = gradcam.generate_heatmap(tensor_img, target_class=true_idx)
        overlay = GradCAM.overlay_heatmap(orig_img, heatmap, alpha=0.45)

        pred_name = idx_to_label[pred_idx]

        # Column 1: Original Image
        axes[i, 0].imshow(orig_img)
        axes[i, 0].set_title(f"Original Leaf\n({variety}, {age}d)", fontsize=10, fontweight="bold")
        axes[i, 0].axis("off")

        # Column 2: Grad-CAM Activation Heatmap
        im_hm = axes[i, 1].imshow(heatmap, cmap="jet")
        axes[i, 1].set_title(f"Grad-CAM Heatmap\nLayer: layer4 (Conv5)", fontsize=10, fontweight="bold")
        axes[i, 1].axis("off")

        # Column 3: Morphological Overlay
        axes[i, 2].imshow(overlay)
        status = "CORRECT" if pred_name == true_label else "MISCLASSIFIED"
        axes[i, 2].set_title(
            f"Salient Feature Overlay\nTrue: {true_label} | Pred: {pred_name} ({conf:.1%})",
            fontsize=10,
            fontweight="bold",
            color="#059669" if status == "CORRECT" else "#dc2626"
        )
        axes[i, 2].axis("off")

    plt.suptitle("Explainable AI (Grad-CAM): Visual Evidence Attribution on Authentic Paddy Leaves", fontsize=12, fontweight="bold", y=0.99)
    plt.tight_layout()
    gradcam_fig_path = figures_dir / "real_gradcam_explanations.png"
    plt.savefig(gradcam_fig_path)
    plt.close()
    print(f"  --> Saved Grad-CAM visualization to: {gradcam_fig_path}")

    # Copy to artifacts directory
    artifact_dir = Path("C:/Users/saiko/.gemini/antigravity-ide/brain/9cd21ae8-d10a-4852-8515-6f8b70aa13ae")
    if artifact_dir.exists():
        import shutil
        shutil.copy2(fig_chart_path, artifact_dir / "real_per_class_f1_comparison.png")
        shutil.copy2(gradcam_fig_path, artifact_dir / "real_gradcam_explanations.png")
        print("  --> Copied visualization artifacts to IDE artifact directory")

    print("[All Real Research Outputs Generated Successfully]")


if __name__ == "__main__":
    main()
