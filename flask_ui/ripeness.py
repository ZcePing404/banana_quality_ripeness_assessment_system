"""Flask bridge for the segmented Teachable Machine ripeness classifier."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from machine_learning.ripeness_classification.predictor import (
    load_ripeness_model,
    predict_ripeness,
)
from machine_learning.ripeness_classification.segmentation import (
    apply_banana_mask,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEGMENTED_MODEL_DIR = (
    PROJECT_ROOT
    / "machine_learning"
    / "ripeness_classification"
    / "models"
    / "segmented"
)

MODEL_PATH = SEGMENTED_MODEL_DIR / "keras_model.h5"
LABELS_PATH = SEGMENTED_MODEL_DIR / "labels.txt"


@lru_cache(maxsize=1)
def _load_cached_model(model_path: str, labels_path: str):
    """Load the H5 model only once for the Flask process."""
    return load_ripeness_model(model_path, labels_path)


def classify_ripeness(
    image_bgr: np.ndarray,
    banana_mask: np.ndarray,
) -> dict[str, object]:
    """Classify ripeness using the SAME banana mask as surface analysis.

    Flask pipeline:

        Original image
            ↓
        Shared K-means banana mask from analyse_image()
            ↓
        Apply that mask and replace background with white
            ↓
        Segmented Teachable Machine model
            ↓
        Unripe / Ripe / Overripe
    """

    if not MODEL_PATH.exists():
        return {
            "available": False,
            "error": f"Segmented ripeness model was not found at: {MODEL_PATH}",
        }

    if not LABELS_PATH.exists():
        return {
            "available": False,
            "error": f"Segmented model labels were not found at: {LABELS_PATH}",
        }

    if image_bgr is None or image_bgr.size == 0:
        return {
            "available": False,
            "error": "Ripeness classification received an empty image.",
        }

    if banana_mask is None or np.asarray(banana_mask).size == 0:
        return {
            "available": False,
            "error": "Ripeness classification received an empty banana mask.",
        }

    try:
        segmented_image, shared_mask = apply_banana_mask(
            image_bgr,
            banana_mask,
        )

        model, labels = _load_cached_model(
            str(MODEL_PATH.resolve()),
            str(LABELS_PATH.resolve()),
        )

        prediction = predict_ripeness(
            segmented_image,
            model,
            labels,
        )

        return {
            "available": True,
            "model_path": str(MODEL_PATH),
            "segmented_image": segmented_image,
            "banana_mask": shared_mask,
            **prediction,
        }

    except Exception as exc:
        return {
            "available": False,
            "error": f"Ripeness classification could not run: {exc}",
        }
