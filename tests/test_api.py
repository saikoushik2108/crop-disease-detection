import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import torch
from api.main import load_artifacts, health_check, MODEL_STATE, ADVISORY_DATABASE


def test_api_startup_and_health():
    """Verify that artifacts and models load properly and health check responds correctly."""
    load_artifacts()
    assert MODEL_STATE["config"] is not None
    assert len(MODEL_STATE["class_names"]) == 10
    assert MODEL_STATE["model"] is not None

    h = health_check()
    assert h.status == "healthy"
    assert h.model_loaded is True
    assert h.num_classes == 10
    assert "bacterial_panicle_blight" in h.classes
    assert "blast" in h.classes


def test_advisory_database_completeness():
    """Verify that every disease class has a valid treatment advisory."""
    classes = [
        "bacterial_leaf_blight",
        "bacterial_leaf_streak",
        "bacterial_panicle_blight",
        "blast",
        "brown_spot",
        "dead_heart",
        "downy_mildew",
        "hispa",
        "normal",
        "tungro"
    ]
    for c in classes:
        assert c in ADVISORY_DATABASE
        assert len(ADVISORY_DATABASE[c]) > 10
