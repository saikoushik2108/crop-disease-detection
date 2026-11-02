import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import io
from PIL import Image
from fastapi import UploadFile, HTTPException
from api.main import health_check, predict_disease, engine, ADVISORY_DATABASE


@pytest.mark.anyio
async def test_api_health_endpoint():
    """Verify that the health check endpoint returns healthy status and metadata."""
    res = await health_check()
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


@pytest.mark.anyio
async def test_predict_invalid_image_rejection():
    """Verify that uploading a non-leaf image returns 400 Bad Request with INVALID_LEAF_IMAGE detail."""
    white_img = Image.new("RGB", (200, 200), color=(250, 250, 250))
    buf = io.BytesIO()
    white_img.save(buf, format="JPEG")
    buf.seek(0)

    upload = UploadFile(filename="white.jpg", file=buf)

    with pytest.raises(HTTPException) as exc_info:
        await predict_disease(image=upload, variety="BPT5204", age=65.0)

    assert exc_info.value.status_code == 400
    detail = exc_info.value.detail
    assert detail["error_type"] == "INVALID_LEAF_IMAGE"
    assert "paddy leaf" in detail["message"]
    assert "వరి ఆకు" in detail["message_te"]


