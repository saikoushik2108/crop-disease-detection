import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.utils.config import load_config, set_seed
from src.preprocessing.image_transforms import get_train_transforms, get_eval_transforms
from src.preprocessing.tabular_pipeline import TabularPreprocessor
from src.models.baseline_image import ImageBaselineResNet
from src.models.baseline_tabular import TabularContextMLP
from src.models.multimodal_net import MultimodalCropNet
from src.evaluation.metrics import compute_classification_metrics


@pytest.fixture
def sample_dataframe():
    """Create miniature mock dataframe for testing matching authentic metadata schema."""
    return pd.DataFrame({
        "image_id": ["10001.jpg", "10002.jpg", "10003.jpg", "10004.jpg"],
        "label": ["blast", "brown_spot", "normal", "blast"],
        "variety": ["ADT37", "ADT43", "Ponni", "ADT37"],
        "age": [45, 60, 75, 50]
    })


def test_config_loading():
    config = load_config("configs/config.yaml")
    assert "project" in config
    assert "dataset" in config
    assert config["dataset"]["num_classes"] == 10
    assert len(config["dataset"]["classes"]) == 10


def test_image_transforms():
    eval_trans = get_eval_transforms(image_size=224)
    dummy_img = Image.new("RGB", (300, 300), color=(100, 150, 80))
    tensor = eval_trans(dummy_img)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_tabular_preprocessor(sample_dataframe):
    preprocessor = TabularPreprocessor()
    var_idx, cont_features = preprocessor.fit_transform(sample_dataframe)
    
    assert len(var_idx) == len(sample_dataframe)
    assert cont_features.shape[0] == len(sample_dataframe)
    assert preprocessor.num_varieties >= 4  # <unk> + 3 unique varieties
    assert preprocessor.num_continuous_features == 1  # age

    # Test unknown variety handling
    test_unseen = pd.DataFrame([{
        "variety": "UNKNOWN_CULTIVAR",
        "age": 55
    }])
    unseen_var, unseen_cont = preprocessor.transform(test_unseen)
    assert unseen_var[0] == preprocessor.variety_to_idx["<unk>"]
    assert unseen_cont.shape == (1, 1)


def test_image_baseline_model():
    model = ImageBaselineResNet(num_classes=10, pretrained=False)
    dummy_x = torch.randn(2, 3, 224, 224)
    logits = model(dummy_x)
    assert logits.shape == (2, 10)
    features = model.extract_features(dummy_x)
    assert features.shape == (2, 512)


def test_tabular_baseline_model():
    model = TabularContextMLP(
        num_varieties=15,
        num_continuous=5,
        num_classes=10,
        variety_embedding_dim=16,
        continuous_proj_dim=16,
        tabular_embedding_dim=64
    )
    dummy_var = torch.tensor([1, 2], dtype=torch.long)
    dummy_cont = torch.randn(2, 5)
    logits = model(dummy_var, dummy_cont)
    assert logits.shape == (2, 10)
    features = model.extract_features(dummy_var, dummy_cont)
    assert features.shape == (2, 64)


def test_multimodal_model():
    model = MultimodalCropNet(
        num_classes=10,
        num_varieties=15,
        num_continuous=5,
        pretrained=False
    )
    dummy_img = torch.randn(2, 3, 224, 224)
    dummy_var = torch.tensor([1, 2], dtype=torch.long)
    dummy_cont = torch.randn(2, 5)
    
    logits = model(dummy_img, dummy_var, dummy_cont)
    assert logits.shape == (2, 10)
    
    f_img, f_tab, f_fused = model.extract_multimodal_features(dummy_img, dummy_var, dummy_cont)
    assert f_img.shape == (2, 512)
    assert f_tab.shape == (2, 64)
    assert f_fused.shape == (2, 576)


def test_metrics_calculation():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 1])
    class_names = ["class_0", "class_1", "class_2"]
    metrics = compute_classification_metrics(y_true, y_pred, class_names)
    
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert metrics["accuracy"] == round(4 / 6, 4)
    assert len(metrics["confusion_matrix_normalized"]) == 3
