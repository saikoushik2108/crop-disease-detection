import os
import random
import yaml
import numpy as np
import torch
from pathlib import Path


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration dictionary from YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int = 42) -> None:
    """Set random seed across all libraries for deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """Detect available compute device (CUDA GPU, Apple MPS, or CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[Device] Utilizing CUDA GPU: {gpu_name}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Device] Utilizing Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        print("[Device] Utilizing CPU")
    return device
