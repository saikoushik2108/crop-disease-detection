import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
import cv2
import glob
from src.preprocessing.leaf_validator import validate_leaf_image


def test_valid_paddy_leaf_images():
    """Verify that authentic paddy leaf photos from the dataset are accepted."""
    sample_images = glob.glob("data/raw/train_images/*/*.jpg")[:15]
    assert len(sample_images) > 0, "No sample images found in dataset."

    for img_path in sample_images:
        is_valid, msg_en, msg_te, metrics = validate_leaf_image(img_path)
        assert is_valid is True, f"Authentic image {img_path} was incorrectly rejected: {msg_en}"
        assert metrics["leaf_ratio"] >= 0.18


def test_blank_and_solid_color_rejection():
    """Verify that blank or solid color non-leaf images are rejected."""
    # Solid black
    black = np.zeros((200, 200, 3), dtype=np.uint8)
    is_valid, msg_en, msg_te, _ = validate_leaf_image(black)
    assert is_valid is False
    assert len(msg_en) > 0 and len(msg_te) > 0

    # Solid white document
    white = np.ones((200, 200, 3), dtype=np.uint8) * 245
    is_valid, msg_en, msg_te, _ = validate_leaf_image(white)
    assert is_valid is False


def test_text_document_grayscale_rejection():
    """Verify that black-and-white text documents or grayscale photos are rejected."""
    doc = np.full((300, 300, 3), 240, dtype=np.uint8)
    # Add fake printed text lines
    for y in range(20, 280, 15):
        cv2.line(doc, (30, y), (270, y), (30, 30, 30), 2)

    is_valid, msg_en, msg_te, _ = validate_leaf_image(doc)
    assert is_valid is False
    assert "black-and-white" in msg_en.lower() or "grayscale" in msg_en.lower() or "document" in msg_en.lower()


def test_unnatural_color_dominance_rejection():
    """Verify that images dominated by blue (sky, cars, clothing) are rejected."""
    blue_car = np.full((250, 250, 3), (220, 120, 20), dtype=np.uint8)  # High BGR blue
    # Add texture noise to exceed std_dev threshold
    noise = np.random.randint(0, 30, (250, 250, 3)).astype(np.uint8)
    blue_car = cv2.add(blue_car, noise)
    # Add road
    blue_car[180:, :] = (80, 80, 80)
    is_valid, msg_en, msg_te, metrics = validate_leaf_image(blue_car)
    assert is_valid is False
    assert "blue" in msg_en.lower() or "foliage" in msg_en.lower() or "plant" in msg_en.lower()


def test_human_portrait_rejection():
    """Verify that human selfies or portraits with dominant skin tones are rejected."""
    portrait = np.full((300, 300, 3), (130, 160, 220), dtype=np.uint8)  # Skin tone BGR
    # Add hair and facial features
    portrait[:70, :] = (20, 20, 20)
    portrait[120:135, 80:110] = (20, 20, 20)
    portrait[120:135, 190:220] = (20, 20, 20)
    portrait[210:225, 120:180] = (50, 50, 180)

    is_valid, msg_en, msg_te, _ = validate_leaf_image(portrait)
    assert is_valid is False
    assert "skin" in msg_en.lower() or "portrait" in msg_en.lower() or "human" in msg_en.lower()


def test_canvas_preset_sample_acceptance():
    """Verify that the frontend's synthetic canvas preset leaf is validated as authentic."""
    canvas = np.zeros((640, 480, 3), dtype=np.uint8)
    canvas[:] = (30, 58, 30)  # Dark green background
    cv2.ellipse(canvas, (240, 320), (140, 300), 0, 0, 360, (51, 122, 59), -1)  # Leaf blade
    # Lesions
    for i in range(5):
        cv2.ellipse(canvas, (200 + i * 10, 180 + i * 45), (12, 28), 12, 0, 360, (64, 113, 168), -1)

    is_valid, msg_en, msg_te, metrics = validate_leaf_image(canvas)
    assert is_valid is True
    assert metrics["leaf_ratio"] > 0.5
