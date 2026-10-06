import argparse
import json
import shutil
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.config import load_config, set_seed, get_device
from src.data.dataset import MultimodalCropDataset
from src.preprocessing.image_transforms import get_train_transforms, get_eval_transforms
from src.preprocessing.tabular_pipeline import TabularPreprocessor
from src.models.baseline_image import ImageBaselineResNet
from src.models.baseline_tabular import TabularContextMLP
from src.models.multimodal_net import MultimodalCropNet
from src.training.trainer import ModelTrainer
from src.evaluation.metrics import (
    compute_classification_metrics,
    save_confusion_matrix_plot,
    save_metrics_to_json
)
from src.evaluation.gradcam import GradCAM
from PIL import Image


def compute_class_weights(train_df: pd.DataFrame, class_names: list, device: torch.device) -> torch.Tensor:
    """Compute inverse frequency class weights strictly from training partition."""
    train_counts = train_df["label"].value_counts()
    total_samples = len(train_df)
    n_classes = len(class_names)
    raw_weights = [total_samples / (n_classes * train_counts.get(c, 1)) for c in class_names]
    weights = torch.tensor(raw_weights, dtype=torch.float32)
    # Normalize weights so sum equals n_classes
    norm_weights = (weights / weights.sum()) * n_classes
    print(f"\n[Class Weights Initialized (Inverse Frequency, Normalized)]")
    for name, w in zip(class_names, norm_weights.tolist()):
        print(f"  {name:<26}: {w:.4f}")
    return norm_weights.to(device)


def run_experiment(
    exp_name: str,
    model: nn.Module,
    mode: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    class_names: list,
    device: torch.device,
    config: dict,
    class_weights: Optional[torch.Tensor] = None,
    checkpoint_dir: str = "models/checkpoints",
    final_model_name: Optional[str] = None
) -> Dict[str, Any]:
    """Train, validate, checkpoint, and evaluate a model configuration."""
    print(f"\n{'='*70}")
    print(f"  STARTING EXPERIMENT: {exp_name.upper()} (Mode: {mode})")
    print(f"{'='*70}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=config["training"]["lr_patience"],
        factor=config["training"]["lr_factor"]
    )

    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        mode=mode,
        checkpoint_dir=checkpoint_dir,
        model_name=exp_name,
        early_stopping_patience=config["training"]["early_stopping_patience"],
        lr_scheduler=scheduler
    )

    # Train
    trainer.fit(num_epochs=config["training"]["num_epochs"])

    # Load best checkpoint for final test evaluation
    best_ckpt = Path(checkpoint_dir) / f"{exp_name}_best.pt"
    if best_ckpt.exists():
        ckpt_data = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_data["model_state_dict"])
        print(f"[Loaded Best Checkpoint] From epoch {ckpt_data.get('epoch', '?')} (Val F1: {ckpt_data.get('val_macro_f1', '?')}) for Test Evaluation")

        # Copy to primary models/ path if requested
        if final_model_name:
            dest_path = Path("models") / f"{final_model_name}.pt"
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_ckpt, dest_path)
            print(f"  --> Copied best model checkpoint to: {dest_path}")

    # Final Test Set Evaluation
    test_loss, test_acc, test_f1, test_preds, test_labels = trainer.evaluate(test_loader)
    metrics = compute_classification_metrics(test_labels, test_preds, class_names)
    metrics["test_loss"] = round(test_loss, 4)

    # Save metrics JSON in results and research
    metrics_path = Path(config["paths"]["metrics_dir"]) / f"{exp_name}_test_metrics.json"
    save_metrics_to_json(metrics, str(metrics_path))

    # Save confusion matrix plot to results and research/figures
    cm_norm = np.array(metrics["confusion_matrix_normalized"])
    cm_path_results = Path(config["paths"]["confusion_matrices_dir"]) / f"{exp_name}_confusion_matrix.png"
    save_confusion_matrix_plot(cm_norm, class_names, str(cm_path_results), title=f"Confusion Matrix: {exp_name}")

    cm_path_research = Path("research/figures") / f"{exp_name}_confusion_matrix.png"
    cm_path_research.parent.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix_plot(cm_norm, class_names, str(cm_path_research), title=f"Normalized Confusion Matrix: {exp_name}")

    print(f"\n[Test Set Results: {exp_name}]")
    print(f"  Accuracy:       {metrics['accuracy']:.2%}")
    print(f"  Macro F1:       {metrics['macro_f1']:.4f}")
    print(f"  Macro Precision:{metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:   {metrics['macro_recall']:.4f}")
    print(f"  Weighted F1:    {metrics['weighted_f1']:.4f}")
    return metrics


def main(config_path: str = "configs/config.yaml", target_exp: str = "all", epochs: Optional[int] = None):
    config = load_config(config_path)
    set_seed(config["project"]["seed"])
    device = get_device()
    torch.set_num_threads(16)

    if epochs is not None:
        config["training"]["num_epochs"] = epochs

    # Load splits
    splits_dir = Path(config["paths"]["splits_dir"])
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")
    test_df = pd.read_csv(splits_dir / "test.csv")

    class_names = config["dataset"]["classes"]
    label_to_idx = {name: i for i, name in enumerate(class_names)}

    # Save class mapping
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    with open(models_dir / "class_mapping.json", "w") as f:
        json.dump(label_to_idx, f, indent=2)

    # Load preprocessor
    preproc_path = Path(config["paths"]["processed_dir"]) / "tabular_preprocessor.json"
    preprocessor = TabularPreprocessor.load(str(preproc_path))
    
    # Save variety mapping and preprocessor in models/
    with open(models_dir / "variety_mapping.json", "w") as f:
        json.dump(preprocessor.variety_to_idx, f, indent=2)
    shutil.copy2(preproc_path, models_dir / "tabular_preprocessor.json")

    # Compute inverse frequency class weights from train_df strictly
    class_weights = compute_class_weights(train_df, class_names, device)

    # Transform tabular features
    train_var, train_cont = preprocessor.transform(train_df)
    val_var, val_cont = preprocessor.transform(val_df)
    test_var, test_cont = preprocessor.transform(test_df)

    # Image transforms
    train_trans = get_train_transforms(config["dataset"]["image_size"])
    eval_trans = get_eval_transforms(config["dataset"]["image_size"])
    image_dir = Path(config["paths"]["raw_dir"]) / "train_images"

    batch_size = config["training"]["batch_size"]
    num_workers = config["project"]["num_workers"]

    # Datasets
    train_ds_img = MultimodalCropDataset(train_df, image_dir, train_var, train_cont, label_to_idx, train_trans, mode="image_only")
    val_ds_img = MultimodalCropDataset(val_df, image_dir, val_var, val_cont, label_to_idx, eval_trans, mode="image_only")
    test_ds_img = MultimodalCropDataset(test_df, image_dir, test_var, test_cont, label_to_idx, eval_trans, mode="image_only")

    train_ds_tab = MultimodalCropDataset(train_df, image_dir, train_var, train_cont, label_to_idx, None, mode="tabular_only")
    val_ds_tab = MultimodalCropDataset(val_df, image_dir, val_var, val_cont, label_to_idx, None, mode="tabular_only")
    test_ds_tab = MultimodalCropDataset(test_df, image_dir, test_var, test_cont, label_to_idx, None, mode="tabular_only")

    train_ds_multi = MultimodalCropDataset(train_df, image_dir, train_var, train_cont, label_to_idx, train_trans, mode="multimodal")
    val_ds_multi = MultimodalCropDataset(val_df, image_dir, val_var, val_cont, label_to_idx, eval_trans, mode="multimodal")
    test_ds_multi = MultimodalCropDataset(test_df, image_dir, test_var, test_cont, label_to_idx, eval_trans, mode="multimodal")

    # Loaders
    train_loader_img = DataLoader(train_ds_img, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader_img = DataLoader(val_ds_img, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader_img = DataLoader(test_ds_img, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    train_loader_tab = DataLoader(train_ds_tab, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader_tab = DataLoader(val_ds_tab, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader_tab = DataLoader(test_ds_tab, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    train_loader_multi = DataLoader(train_ds_multi, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader_multi = DataLoader(val_ds_multi, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader_multi = DataLoader(test_ds_multi, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Save model config
    model_cfg = {
        "num_classes": len(class_names),
        "num_varieties": preprocessor.num_varieties,
        "num_continuous": preprocessor.num_continuous_features,
        "classes": class_names,
        "image_size": config["dataset"]["image_size"],
        "variety_embedding_dim": config["models"]["variety_embedding_dim"],
        "continuous_feature_dim": config["models"]["continuous_feature_dim"],
        "tabular_embedding_dim": config["models"]["tabular_embedding_dim"],
        "fusion_hidden_dim": config["models"]["fusion_hidden_dim"],
        "dropout": config["models"]["dropout"]
    }
    with open(models_dir / "model_config.json", "w") as f:
        json.dump(model_cfg, f, indent=2)

    results = {}

    # =========================================================================
    # EXPERIMENT B: Metadata-Only Baseline (Variety + Age MLP)
    # =========================================================================
    if target_exp in ["b", "metadata_only", "all"]:
        model_tab = TabularContextMLP(
            num_varieties=preprocessor.num_varieties,
            num_continuous=preprocessor.num_continuous_features,
            num_classes=len(class_names),
            variety_embedding_dim=config["models"]["variety_embedding_dim"],
            continuous_proj_dim=config["models"]["continuous_feature_dim"],
            tabular_embedding_dim=config["models"]["tabular_embedding_dim"],
            dropout=config["models"]["dropout"] * 0.75
        )
        res_b = run_experiment(
            exp_name="metadata_only",
            model=model_tab,
            mode="tabular_only",
            train_loader=train_loader_tab,
            val_loader=val_loader_tab,
            test_loader=test_loader_tab,
            class_names=class_names,
            device=device,
            config=config,
            class_weights=class_weights,
            final_model_name="metadata_only_best"
        )
        results["Metadata Only"] = res_b

    # =========================================================================
    # EXPERIMENT A: Image-Only Baseline (ResNet18 Transfer Learning)
    # =========================================================================
    if target_exp in ["a", "image_only", "all"]:
        model_img = ImageBaselineResNet(
            num_classes=len(class_names),
            pretrained=config["models"]["pretrained"],
            dropout=config["models"]["dropout"]
        )
        res_a = run_experiment(
            exp_name="image_only",
            model=model_img,
            mode="image_only",
            train_loader=train_loader_img,
            val_loader=val_loader_img,
            test_loader=test_loader_img,
            class_names=class_names,
            device=device,
            config=config,
            class_weights=class_weights,
            final_model_name="image_only_best"
        )
        results["Image Only"] = res_a

    # =========================================================================
    # EXPERIMENT C: Multimodal Fusion (ResNet18 + Variety + Age)
    # =========================================================================
    if target_exp in ["c", "multimodal", "all"]:
        model_multi = MultimodalCropNet(
            num_classes=len(class_names),
            num_varieties=preprocessor.num_varieties,
            num_continuous=preprocessor.num_continuous_features,
            pretrained=config["models"]["pretrained"],
            variety_embedding_dim=config["models"]["variety_embedding_dim"],
            continuous_proj_dim=config["models"]["continuous_feature_dim"],
            tabular_embedding_dim=config["models"]["tabular_embedding_dim"],
            fusion_hidden_dim=config["models"]["fusion_hidden_dim"],
            dropout=config["models"]["dropout"]
        )
        res_c = run_experiment(
            exp_name="multimodal",
            model=model_multi,
            mode="multimodal",
            train_loader=train_loader_multi,
            val_loader=val_loader_multi,
            test_loader=test_loader_multi,
            class_names=class_names,
            device=device,
            config=config,
            class_weights=class_weights,
            final_model_name="multimodal_best"
        )
        results["Multimodal"] = res_c

    print(f"\n[Execution Completed for Target: {target_exp}]")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--experiment", type=str, default="all", choices=["a", "b", "c", "image_only", "metadata_only", "multimodal", "all"])
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    args = parser.parse_args()
    main(config_path=args.config, target_exp=args.experiment, epochs=args.epochs)
