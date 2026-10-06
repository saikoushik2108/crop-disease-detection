import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm


class ModelTrainer:
    """
    Universal PyTorch Trainer supporting:
    - Image-Only Baseline
    - Tabular-Only Baseline
    - Multimodal Fusion Network
    
    Includes Early Stopping, Learning Rate Scheduling, Checkpointing,
    and History Tracking.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        mode: str = "multimodal",
        checkpoint_dir: str = "models/checkpoints",
        model_name: str = "model",
        early_stopping_patience: int = 4,
        lr_scheduler: Optional[Any] = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.mode = mode
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.patience = early_stopping_patience
        self.scheduler = lr_scheduler

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "val_macro_f1": []
        }
        self.best_val_f1 = -1.0
        self.best_epoch = -1
        self.epochs_without_improvement = 0

    def _forward_batch(self, batch_data: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Execute forward pass according to active modality mode."""
        if self.mode == "image_only":
            images, labels = batch_data
            images = images.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(images)
            return logits, labels

        elif self.mode == "tabular_only":
            (variety_idx, cont_features), labels = batch_data
            variety_idx = variety_idx.to(self.device)
            cont_features = cont_features.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(variety_idx, cont_features)
            return logits, labels

        elif self.mode == "multimodal":
            (images, variety_idx, cont_features), labels = batch_data
            images = images.to(self.device)
            variety_idx = variety_idx.to(self.device)
            cont_features = cont_features.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(images, variety_idx, cont_features)
            return logits, labels

        else:
            raise ValueError(f"Unsupported modality mode: {self.mode}")

    def train_epoch(self) -> Tuple[float, float]:
        """Train model for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        num_batches = len(self.train_loader)
        for batch_idx, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            logits, labels = self._forward_batch(batch)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == num_batches:
                batch_acc = correct / total if total > 0 else 0.0
                print(f"    Batch [{batch_idx + 1:03d}/{num_batches:03d}] - Loss: {loss.item():.4f} - Running Acc: {batch_acc:.2%}", flush=True)

        epoch_loss = running_loss / total if total > 0 else 0.0
        epoch_acc = correct / total if total > 0 else 0.0
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def evaluate(self, loader: Optional[DataLoader] = None) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
        """Evaluate model returning loss, accuracy, macro-F1, all preds, all true labels."""
        if loader is None:
            loader = self.val_loader

        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for batch in loader:
            logits, labels = self._forward_batch(batch)
            loss = self.criterion(logits, labels)
            running_loss += loss.item() * len(labels)

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        total = len(all_labels)
        val_loss = running_loss / total if total > 0 else 0.0
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        acc = (all_preds == all_labels).mean() if total > 0 else 0.0
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        return val_loss, acc, macro_f1, all_preds, all_labels

    def fit(self, num_epochs: int) -> Dict[str, List[float]]:
        """Run full training loop with early stopping and checkpointing."""
        print(f"\n[Starting Training: {self.model_name}] (Modality: {self.mode})")
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, val_f1, _, _ = self.evaluate()
            elapsed = time.time() - start_time

            self.history["train_loss"].append(round(train_loss, 4))
            self.history["val_loss"].append(round(val_loss, 4))
            self.history["train_acc"].append(round(train_acc, 4))
            self.history["val_acc"].append(round(val_acc, 4))
            self.history["val_macro_f1"].append(round(val_f1, 4))

            print(
                f"Epoch [{epoch:02d}/{num_epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3%} || "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3%} | Val Macro F1: {val_f1:.4f} "
                f"({elapsed:.1f}s)"
            )

            # Learning rate scheduler step
            if self.scheduler is not None:
                self.scheduler.step(val_f1)

            # Checkpoint on best Validation Macro F1
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                ckpt_path = self.checkpoint_dir / f"{self.model_name}_best.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_macro_f1": val_f1,
                    "val_acc": val_acc
                }, ckpt_path)
                print(f"  --> Saved new best checkpoint: {ckpt_path} (F1: {val_f1:.4f})")
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    print(f"\n[Early Stopping Triggered] No improvement in validation F1 for {self.patience} epochs.")
                    break

        print(f"[Training Completed] Best Validation F1: {self.best_val_f1:.4f} at Epoch {self.best_epoch}")
        self._plot_history()
        return self.history

    def _plot_history(self) -> None:
        """Plot and save training and validation curves."""
        fig_path = Path("results/figures") / f"{self.model_name}_learning_curves.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        res_fig_path = Path("research/figures") / f"{self.model_name}_training_curve.png"
        res_fig_path.parent.mkdir(parents=True, exist_ok=True)

        epochs = range(1, len(self.history["train_loss"]) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        # Loss curves
        ax1.plot(epochs, self.history["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
        ax1.plot(epochs, self.history["val_loss"], label="Val Loss", color="#ff7f0e", lw=2)
        ax1.set_title(f"{self.model_name} - Cross-Entropy Loss", fontweight="bold")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend()

        # Accuracy & F1 curves
        ax2.plot(epochs, self.history["train_acc"], label="Train Acc", color="#2ca02c", lw=2)
        ax2.plot(epochs, self.history["val_acc"], label="Val Acc", color="#d62728", lw=2)
        ax2.plot(epochs, self.history["val_macro_f1"], label="Val Macro F1", color="#9467bd", lw=2, linestyle=":")
        ax2.set_title(f"{self.model_name} - Accuracy & Macro F1", fontweight="bold")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Metric Score")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.savefig(res_fig_path, dpi=300)
        plt.close()
