import os
from pathlib import Path
from typing import Tuple, Dict
import pandas as pd
from sklearn.model_selection import train_test_split


def create_stratified_splits(
    metadata_df: pd.DataFrame,
    label_col: str = "label",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    output_dir: str = "data/processed/splits"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split metadata into stratified Train, Validation, and Test sets.
    
    Guarantees:
    - Zero image leakage between splits.
    - Proportional representation of all disease classes across splits.
    - Deterministic reproducibility using specified random seed.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Split ratios must sum to 1.0"

    # First split: Train vs (Val + Test)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        metadata_df,
        test_size=temp_ratio,
        stratify=metadata_df[label_col],
        random_state=seed
    )

    # Second split: Val vs Test
    relative_test_ratio = test_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        stratify=temp_df[label_col],
        random_state=seed
    )

    # Leakage check: Verify disjoint image_ids
    train_ids = set(train_df["image_id"])
    val_ids = set(val_df["image_id"])
    test_ids = set(test_df["image_id"])

    assert len(train_ids.intersection(val_ids)) == 0, "DATA LEAKAGE DETECTED: Train and Val overlap!"
    assert len(train_ids.intersection(test_ids)) == 0, "DATA LEAKAGE DETECTED: Train and Test overlap!"
    assert len(val_ids.intersection(test_ids)) == 0, "DATA LEAKAGE DETECTED: Val and Test overlap!"

    # Save to disk
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(out_path / "train.csv", index=False)
    val_df.to_csv(out_path / "val.csv", index=False)
    test_df.to_csv(out_path / "test.csv", index=False)

    print(f"[Data Splitting Complete]")
    print(f"  Training samples:   {len(train_df)} ({len(train_df)/len(metadata_df):.1%})")
    print(f"  Validation samples: {len(val_df)} ({len(val_df)/len(metadata_df):.1%})")
    print(f"  Test samples:       {len(test_df)} ({len(test_df)/len(metadata_df):.1%})")
    
    return train_df, val_df, test_df


def inspect_split_distribution(df: pd.DataFrame, split_name: str, label_col: str = "label") -> pd.DataFrame:
    """Return distribution of classes within a split."""
    dist = df[label_col].value_counts().reset_index()
    dist.columns = ["class", f"{split_name}_count"]
    dist[f"{split_name}_pct"] = (dist[f"{split_name}_count"] / len(df) * 100).round(2)
    return dist
