"""
Botanical Leaf Image Validator & Out-of-Domain (OOD) Detector.

Validates whether an uploaded image contains authentic paddy/rice (Oryza sativa)
leaf tissue before passing it to deep learning inference. Rejects out-of-domain images
such as human portraits, vehicles, text documents, solid colors, and arbitrary objects.
"""

from typing import Tuple, Dict, Any, Union
import numpy as np
import cv2
from pathlib import Path


def validate_leaf_image(
    image_input: Union[str, Path, bytes, np.ndarray]
) -> Tuple[bool, str, str, Dict[str, Any]]:
    """
    Validate whether an input image represents a legitimate paddy leaf photograph.

    Parameters:
    -----------
    image_input: str, Path, bytes, or numpy.ndarray
        Filepath, raw bytes, or decoded BGR numpy image.

    Returns:
    --------
    Tuple[bool, str, str, Dict[str, Any]]:
        - is_valid (bool): True if the image is an authentic leaf image, False otherwise.
        - reason_en (str): Explanation in English.
        - reason_te (str): Explanation in Telugu.
        - metrics (dict): Quantitative computer vision metrics extracted from the image.
    """
    # 1. Decode input into BGR NumPy array
    img = None
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
    elif isinstance(image_input, bytes):
        nparr = np.frombuffer(image_input, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif isinstance(image_input, np.ndarray):
        img = image_input

    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        return (
            False,
            "The uploaded file could not be decoded as a valid image.",
            "అప్‌లోడ్ చేసిన ఫైల్ చెల్లుబాటు అయ్యే చిత్రంగా తెరవబడలేదు.",
            {"error": "DECODE_ERROR"}
        )

    h, w = img.shape[:2]
    total_pixels = float(h * w)

    if total_pixels < 2500:  # Less than 50x50
        return (
            False,
            "Image resolution is too low (minimum 50x50 pixels required).",
            "చిత్రం రిజల్యూషన్ చాలా తక్కువగా ఉంది (కనీసం 50x50 పిక్సెల్స్ అవసరం).",
            {"resolution": f"{w}x{h}"}
        )

    # 2. Check for solid color / blank / extreme low variance
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    std_dev = float(np.std(gray))
    if std_dev < 10.0:
        return (
            False,
            "Image appears to be blank, completely dark, or a solid color.",
            "చిత్రం ఖాళీగా, పూర్తిగా చీకటిగా లేదా ఒకే రంగుతో కనిపిస్తోంది.",
            {"std_dev": round(std_dev, 2)}
        )

    # 3. Color Space Analysis (HSV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    mean_sat = float(np.mean(sat))
    mean_val = float(np.mean(val))

    # Reject grayscale, text documents, or B&W papers
    if mean_sat < 20.0 and mean_val > 35.0:
        return (
            False,
            "Image appears to be a black-and-white text document, paper, or grayscale capture.",
            "చిత్రం బ్లాక్-అండ్-వైట్ పత్రం, కాగితం లేదా గ్రేస్కేల్‌గా కనిపిస్తోంది.",
            {"mean_sat": round(mean_sat, 2), "mean_val": round(mean_val, 2)}
        )

    # 4. Agro-Botanical Foliage & Lesion Segmentation
    # Foliage green & yellow-green: H in [22, 88], S >= 25, V >= 25
    foliage_green = (hue >= 22) & (hue <= 88) & (sat >= 25) & (val >= 25)
    # Chlorotic yellow / tan / brown lesions: H in [10, 24], S >= 35, V >= 25
    lesion_brown = (hue >= 10) & (hue < 24) & (sat >= 35) & (val >= 25)
    # Necrotic dark lesion edges: H in [5, 10], S >= 40, V in [20, 160]
    necrotic = (hue >= 5) & (hue < 10) & (sat >= 40) & (val >= 20) & (val <= 160)

    leaf_mask = foliage_green | lesion_brown | necrotic
    leaf_ratio = float(np.sum(leaf_mask) / total_pixels)
    green_ratio = float(np.sum(foliage_green) / total_pixels)

    # 5. Non-Vegetation Color Dominance Checks
    # Unnatural blues, cyans, magentas (sky, vehicles, clothing, screens)
    blue_cyan_magenta = (hue >= 92) & (hue <= 165) & (sat >= 40) & (val >= 30)
    blue_ratio = float(np.sum(blue_cyan_magenta) / total_pixels)
    if blue_ratio > 0.35:
        return (
            False,
            f"Image is dominated by non-plant background colors (sky, blue vehicle, or clothing: {blue_ratio:.1%}).",
            f"చిత్రంలో ఆకులకు చెందని రంగులు (ఆకాశం, నీలి వాహనం లేదా దుస్తులు: {blue_ratio:.1%}) ఎక్కువగా ఉన్నాయి.",
            {"blue_ratio": round(blue_ratio, 4), "leaf_ratio": round(leaf_ratio, 4)}
        )

    # Human skin tones dominance (portraits, selfies, hands with no leaf)
    skin_tone = (hue <= 16) & (sat >= 25) & (sat <= 170) & (val >= 60)
    skin_ratio = float(np.sum(skin_tone) / total_pixels)
    if skin_ratio > 0.40 and green_ratio < 0.15:
        return (
            False,
            f"Image appears to be a human portrait or selfie rather than a crop leaf ({skin_ratio:.1%} skin tones).",
            f"చిత్రం పంట ఆకు కాకుండా మనిషి ముఖం లేదా పోర్ట్రెయిట్‌గా కనిపిస్తోంది ({skin_ratio:.1%} స్కిన్ టోన్స్).",
            {"skin_ratio": round(skin_ratio, 4), "green_ratio": round(green_ratio, 4)}
        )

    # 6. Agronomic Excess Green Index (ExG = 2*G - R - B)
    b_ch, g_ch, r_ch = cv2.split(img.astype(np.float32))
    exg = 2.0 * g_ch - r_ch - b_ch
    exg_positive_ratio = float(np.sum(exg > 5.0) / total_pixels)

    # 7. Final Botanical Threshold Validation
    # Real paddy leaf photos have substantial foliage/lesion area (typically >50%, thresholded at 18%)
    if leaf_ratio < 0.18 and exg_positive_ratio < 0.15:
        return (
            False,
            f"No paddy leaf or crop foliage detected (foliage coverage only {leaf_ratio:.1%}). Please upload a clear photo of a rice leaf.",
            f"చిత్రంలో వరి ఆకు లేదా పైరు లక్షణాలు కనిపించలేదు (ఆకు విస్తీర్ణం కేవలం {leaf_ratio:.1%}). దయచేసి వరి ఆకు స్పష్టమైన ఫోటోను అప్‌లోడ్ చేయండి.",
            {
                "leaf_ratio": round(leaf_ratio, 4),
                "green_ratio": round(green_ratio, 4),
                "exg_positive_ratio": round(exg_positive_ratio, 4)
            }
        )

    metrics = {
        "leaf_ratio": round(leaf_ratio, 4),
        "green_ratio": round(green_ratio, 4),
        "exg_positive_ratio": round(exg_positive_ratio, 4),
        "std_dev": round(std_dev, 2),
        "mean_sat": round(mean_sat, 2)
    }

    return (
        True,
        "Authentic paddy leaf image validated.",
        "వరి ఆకు చిత్రం నిర్ధారించబడింది.",
        metrics
    )

