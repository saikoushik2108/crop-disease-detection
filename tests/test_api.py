import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from api.main import health_check, engine, ADVISORY_DATABASE


def test_api_health_endpoint():
    """Verify that the health check endpoint returns healthy status and metadata."""
    res = asyncio.run(health_check())
    assert res["status"] == "healthy"
    assert res["model_loaded"] is True
    assert res["num_classes"] == 10


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


def test_inference_engine_loaded():
    """Verify that the singleton InferenceEngine has loaded the multimodal model."""
    assert engine is not None
    assert engine.model is not None
    assert engine.preprocessor is not None
