import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str]
) -> Dict[str, Any]:
    """
    Compute comprehensive scientific classification metrics.
    
    Returns:
    - Overall Accuracy
    - Macro Precision, Recall, F1
    - Weighted Precision, Recall, F1
    - Per-class metrics dictionary
    - Raw and normalized confusion matrices
    """
    acc = float(accuracy_score(y_true, y_pred))
    macro_p = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_r = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    
    weighted_p = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    weighted_r = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cm_raw = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    # Safe normalization: divide by row sum
    row_sums = cm_raw.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_raw.astype(float), row_sums, out=np.zeros_like(cm_raw, dtype=float), where=row_sums != 0)

    # Per-class breakdown
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    metrics = {
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_precision": round(weighted_p, 4),
        "weighted_recall": round(weighted_r, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": {
            cls_name: {
                "precision": round(report[cls_name]["precision"], 4),
                "recall": round(report[cls_name]["recall"], 4),
                "f1": round(report[cls_name]["f1-score"], 4),
                "support": int(report[cls_name]["support"])
            }
            for cls_name in class_names if cls_name in report
        },
        "confusion_matrix_raw": cm_raw.tolist(),
        "confusion_matrix_normalized": [[round(val, 4) for val in row] for row in cm_norm]
    }
    return metrics


def save_confusion_matrix_plot(
    cm_norm: np.ndarray,
    class_names: List[str],
    output_path: str,
    title: str = "Normalized Confusion Matrix"
) -> None:
    """Render and save publication-grade normalized confusion matrix heatmap."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Normalized Rate"}
    )
    plt.title(title, fontsize=14, pad=12, fontweight="bold")
    plt.xlabel("Predicted Disease Class", fontsize=12, labelpad=10)
    plt.ylabel("True Disease Class", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_metrics_to_json(metrics: Dict[str, Any], output_path: str) -> None:
    """Save metrics dictionary to formatted JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
