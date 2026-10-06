import os
from pathlib import Path
from typing import Callable, Optional, Tuple, Dict, Any, Union
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class MultimodalCropDataset(Dataset):
    """
    PyTorch Dataset supporting:
    1. Visual imagery (leaf RGB)
    2. Agronomic metadata (variety index, plant age)
    3. Regional agro-environmental context features
    
    Operates in three modes:
    - 'multimodal': yields (image, variety_idx, cont_features), label
    - 'image_only': yields image, label
    - 'tabular_only': yields (variety_idx, cont_features), label
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: Union[str, Path],
        variety_indices: np.ndarray,
        cont_features: np.ndarray,
        label_to_idx: Dict[str, int],
        transform: Optional[Callable] = None,
        mode: str = "multimodal"
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.variety_indices = torch.tensor(variety_indices, dtype=torch.long)
        self.cont_features = torch.tensor(cont_features, dtype=torch.float32)
        self.label_to_idx = label_to_idx
        self.transform = transform
        self.mode = mode

        assert len(self.df) == len(self.variety_indices) == len(self.cont_features), \
            "Dataframe and preprocessed tabular arrays must have identical length"

        # Pre-extract image paths and labels
        self.image_filenames = self.df["image_id"].tolist()
        self.raw_labels = self.df["label"].tolist()
        self.labels = [self.label_to_idx[l] for l in self.raw_labels]
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, filename: str, label_name: Optional[str] = None) -> Image.Image:
        """Load RGB image safely with O(1) direct folder lookup."""
        # 1. Direct check in class subdirectory if label_name provided
        if label_name is not None:
            direct_path = self.image_dir / label_name / filename
            if direct_path.exists():
                return Image.open(direct_path).convert("RGB")

        # 2. Direct check in image_dir root
        img_path = self.image_dir / filename
        if img_path.exists():
            return Image.open(img_path).convert("RGB")

        # 3. Check in subdirectories if images are organized by class
        for sub_dir in self.image_dir.iterdir():
            if sub_dir.is_dir() and (sub_dir / filename).exists():
                return Image.open(sub_dir / filename).convert("RGB")

        # 4. Fallback if filename itself is a relative path
        alt_path = Path(filename)
        if alt_path.exists():
            return Image.open(alt_path).convert("RGB")

        raise FileNotFoundError(f"Image not found: {filename} in {self.image_dir}")

    def __getitem__(self, idx: int) -> Tuple[Any, torch.Tensor]:
        label = self.labels[idx]

        if self.mode == "tabular_only":
            variety_idx = self.variety_indices[idx]
            cont_feat = self.cont_features[idx]
            return (variety_idx, cont_feat), label

        # Load image for image_only and multimodal modes
        image = self._load_image(self.image_filenames[idx], self.raw_labels[idx])
        if self.transform is not None:
            image = self.transform(image)

        if self.mode == "image_only":
            return image, label

        # Multimodal mode
        variety_idx = self.variety_indices[idx]
        cont_feat = self.cont_features[idx]
        return (image, variety_idx, cont_feat), label
