import json
import shutil
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.config import load_config, get_device
from src.preprocessing.image_transforms import get_eval_transforms
from src.preprocessing.tabular_pipeline import TabularPreprocessor
from src.data.dataset import MultimodalCropDataset
from src.models.multimodal_net import MultimodalCropNet
from src.evaluation.metrics import (
    compute_classification_metrics,
    save_confusion_matrix_plot,
    save_metrics_to_json,
)


def evaluate_checkpoint():
    config_path = "configs/config.yaml"
    config = load_config(config_path)
    device = get_device()
    torch.set_num_threads(16)

    splits_dir = Path(config["paths"]["splits_dir"])
    test_df = pd.read_csv(splits_dir / "test.csv")
    class_names = config["dataset"]["classes"]
    label_to_idx = {name: i for i, name in enumerate(class_names)}

    preproc_path = Path("models/tabular_preprocessor.json")
    if not preproc_path.exists():
        preproc_path = Path(config["paths"]["processed_dir"]) / "tabular_preprocessor.json"
    preprocessor = TabularPreprocessor.load(str(preproc_path))

    test_var, test_cont = preprocessor.transform(test_df)
    eval_trans = get_eval_transforms(config["dataset"]["image_size"])
    image_dir = Path(config["paths"]["raw_dir"]) / "train_images"

    batch_size = config["training"]["batch_size"]
    test_ds_multi = MultimodalCropDataset(
        test_df, image_dir, test_var, test_cont, label_to_idx, eval_trans, mode="multimodal"
    )
    test_loader_multi = DataLoader(
        test_ds_multi, batch_size=batch_size, shuffle=False, num_workers=config["project"]["num_workers"]
    )

    ckpt_path = Path("models/checkpoints/multimodal_best.pt")
    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return

    print(f"\n[Loading Model Checkpoint: {ckpt_path}]")
    ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)
    epoch = ckpt_data.get("epoch", "?")
    val_macro_f1 = ckpt_data.get("val_macro_f1", "?")
    val_acc = ckpt_data.get("val_acc", "?")
    print(f"  Checkpoint Epoch: {epoch}")
    print(f"  Val Macro F1: {val_macro_f1}")
    print(f"  Val Accuracy: {val_acc}")

    model = MultimodalCropNet(
        num_classes=len(class_names),
        num_varieties=preprocessor.num_varieties,
        num_continuous=preprocessor.num_continuous_features,
        pretrained=False,
        variety_embedding_dim=config["models"]["variety_embedding_dim"],
        continuous_proj_dim=config["models"]["continuous_feature_dim"],
        tabular_embedding_dim=config["models"]["tabular_embedding_dim"],
        fusion_hidden_dim=config["models"]["fusion_hidden_dim"],
        dropout=config["models"]["dropout"]
    )
    model.load_state_dict(ckpt_data["model_state_dict"])
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    print("\n[Running Test Evaluation on 1,561 Real Test Samples...]")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader_multi):
            (images, varieties, continuous), labels = batch
            images = images.to(device)
            varieties = varieties.to(device)
            continuous = continuous.to(device)
            labels = labels.to(device)

            outputs = model(images, varieties, continuous)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    test_loss = total_loss / len(test_loader_multi.dataset)
    metrics = compute_classification_metrics(all_labels, all_preds, class_names)
    metrics["test_loss"] = round(test_loss, 4)

    # Save metrics JSON in results and research
    metrics_path = Path(config["paths"]["metrics_dir"]) / "multimodal_test_metrics.json"
    save_metrics_to_json(metrics, str(metrics_path))
    print(f"  --> Saved test metrics to: {metrics_path}")

    # Copy to models/multimodal_best.pt
    dest_path = Path("models/multimodal_best.pt")
    shutil.copy2(ckpt_path, dest_path)
    print(f"  --> Copied best model checkpoint to: {dest_path}")

    # Confusion matrix plots
    cm_norm = np.array(metrics["confusion_matrix_normalized"])
    cm_path_results = Path(config["paths"]["confusion_matrices_dir"]) / "multimodal_confusion_matrix.png"
    save_confusion_matrix_plot(cm_norm, class_names, str(cm_path_results), title="Confusion Matrix: Multimodal Fusion")

    cm_path_research = Path("research/figures/multimodal_confusion_matrix.png")
    cm_path_research.parent.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix_plot(cm_norm, class_names, str(cm_path_research), title="Normalized Confusion Matrix: Multimodal Fusion")

    print("\n" + "=" * 60)
    print("  MULTIMODAL FUSION TEST SET RESULTS (Real Paddy Doctor Data)")
    print("=" * 60)
    print(f"  Test Loss:        {metrics['test_loss']:.4f}")
    print(f"  Test Accuracy:    {metrics['accuracy']:.2%}")
    print(f"  Macro F1:         {metrics['macro_f1']:.4f}")
    print(f"  Macro Precision:  {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:     {metrics['macro_recall']:.4f}")
    print(f"  Weighted F1:      {metrics['weighted_f1']:.4f}")
    print("=" * 60)

    for cls_name, cls_m in metrics["per_class"].items():
        print(f"  {cls_name:<25}: P={cls_m['precision']:.3f}, R={cls_m['recall']:.3f}, F1={cls_m['f1']:.3f} (N={cls_m['support']})")


if __name__ == "__main__":
    evaluate_checkpoint()
