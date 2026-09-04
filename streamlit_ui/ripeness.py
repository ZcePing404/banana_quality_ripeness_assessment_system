"""Streamlit bridge for the segmented Teachable Machine ripeness classifier."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st

from machine_learning.ripeness_classification.predictor import (
    load_ripeness_model,
    predict_ripeness,
)
from machine_learning.ripeness_classification.segmentation import apply_banana_mask


# ============================================================
# MODEL PATH
# ============================================================

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


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner=False)
def _load_cached_model(
    model_path: str,
    labels_path: str,
):
    """Load the Teachable Machine model once per Streamlit process."""
    return load_ripeness_model(model_path, labels_path)


# ============================================================
# STREAMLIT RIPENESS CLASSIFICATION
# ============================================================

def classify_ripeness_for_ui(
    image_bgr: np.ndarray,
    banana_mask: np.ndarray,
) -> dict[str, object]:
    """Classify ripeness using the SAME banana mask as blemish analysis.

    Streamlit pipeline::

        Original image
            ↓
        Shared K-means banana mask from analyse_image()
        (K follows the current Streamlit slider value)
            ↓
        Replace pixels outside that mask with white
            ↓
        Segmented Teachable Machine model
            ↓
        Unripe / Ripe / Overripe

    No second K-means segmentation and no GrabCut are run here.
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
        # Reuse the final K-means mask already produced for blemish analysis.
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
