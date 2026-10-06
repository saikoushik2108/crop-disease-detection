import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd


class TabularPreprocessor:
    """
    Preprocessor for agronomic and environmental tabular features.
    
    CRITICAL SCIENTIFIC INTEGRITY RULE:
    Statistics (means, standard deviations, categorical encodings) MUST be fit
    strictly on the training split to eliminate data leakage into validation/test splits.
    """
    
    def __init__(self):
        self.variety_to_idx: Dict[str, int] = {}
        self.idx_to_variety: Dict[int, str] = {}
        self.continuous_cols: List[str] = ["age"]
        self.env_cols: List[str] = []
        self.scaler_params: Dict[str, Dict[str, float]] = {}
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        """Fit preprocessor parameters strictly on training dataframe."""
        # 1. Fit categorical mapping for 'variety'
        varieties = sorted(df["variety"].dropna().unique().tolist())
        # Index 0 is reserved for '<unk>' (unknown / unseen varieties)
        self.variety_to_idx = {"<unk>": 0}
        for idx, var in enumerate(varieties, start=1):
            self.variety_to_idx[str(var)] = idx
        self.idx_to_variety = {idx: var for var, idx in self.variety_to_idx.items()}

        # 2. Fit scalers for continuous features (age + environmental)
        target_cols = [c for c in (self.continuous_cols + self.env_cols) if c in df.columns]
        self.scaler_params = {}
        for col in target_cols:
            series = df[col].dropna()
            mean_val = float(series.mean()) if len(series) > 0 else 0.0
            std_val = float(series.std()) if len(series) > 1 and series.std() > 1e-6 else 1.0
            self.scaler_params[col] = {
                "mean": mean_val,
                "std": std_val,
                "median": float(series.median()) if len(series) > 0 else 0.0
            }

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform dataframe into:
        1. variety_indices: np.ndarray of shape (N,) int64
        2. continuous_features: np.ndarray of shape (N, C) float32
        """
        if not self.is_fitted:
            raise RuntimeError("TabularPreprocessor must be fitted on training data before transforming.")

        # 1. Transform variety
        variety_indices = []
        for var in df["variety"].fillna("<unk>"):
            idx = self.variety_to_idx.get(str(var), self.variety_to_idx["<unk>"])
            variety_indices.append(idx)
        variety_array = np.array(variety_indices, dtype=np.int64)

        # 2. Transform continuous & environmental features
        cont_features_list = []
        target_cols = [c for c in (self.continuous_cols + self.env_cols) if c in self.scaler_params]
        
        for col in target_cols:
            vals = df[col].copy() if col in df.columns else pd.Series([np.nan] * len(df))
            median_fill = self.scaler_params[col]["median"]
            vals = vals.fillna(median_fill)
            mean = self.scaler_params[col]["mean"]
            std = self.scaler_params[col]["std"]
            normed = (vals.to_numpy(dtype=np.float32) - mean) / std
            cont_features_list.append(normed)

        if cont_features_list:
            cont_array = np.column_stack(cont_features_list)
        else:
            cont_array = np.zeros((len(df), 1), dtype=np.float32)

        return variety_array, cont_array

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Fit on training df and transform."""
        return self.fit(df).transform(df)

    @property
    def num_varieties(self) -> int:
        """Total number of variety embedding categories including <unk>."""
        return len(self.variety_to_idx)

    @property
    def num_continuous_features(self) -> int:
        """Total number of continuous tabular features."""
        return len(self.scaler_params)

    def save(self, file_path: str) -> None:
        """Save fitted state to JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "variety_to_idx": self.variety_to_idx,
            "continuous_cols": self.continuous_cols,
            "env_cols": self.env_cols,
            "scaler_params": self.scaler_params,
            "is_fitted": self.is_fitted
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, file_path: str) -> "TabularPreprocessor":
        """Load fitted state from JSON file."""
        path = Path(file_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        preprocessor = cls()
        preprocessor.variety_to_idx = data["variety_to_idx"]
        preprocessor.idx_to_variety = {int(v): k for k, v in data["variety_to_idx"].items()}
        preprocessor.continuous_cols = data["continuous_cols"]
        preprocessor.env_cols = data["env_cols"]
        preprocessor.scaler_params = data["scaler_params"]
        preprocessor.is_fitted = data["is_fitted"]
        return preprocessor
