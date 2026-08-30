"""Streamlit bridge for the Teachable Machine ripeness classifier."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st

from machine_learning.ripeness_classification.predictor import (
    find_ripeness_model_files,
    load_ripeness_model,
    predict_ripeness,
)


@st.cache_resource(show_spinner=False)
def _load_cached_model(model_path: str, labels_path: str):
    """Load the relatively large H5 model only once per Streamlit process."""
    return load_ripeness_model(model_path, labels_path)


def classify_ripeness_for_ui(image_bgr: np.ndarray) -> dict[str, object]:
    """Classify one image and return a UI-safe status dictionary.

    The same function is used for uploaded images and frozen USB-camera frames,
    so the camera needs no separate ML path.
    """
    model_path, labels_path = find_ripeness_model_files()

    if model_path is None or labels_path is None:
        return {
            "available": False,
            "error": (
                "Ripeness model files were not found. Put your Teachable Machine "
                ".h5 model and labels.txt in "
                "machine_learning/ripeness_classification/models/."
            ),
        }

    try:
        model, labels = _load_cached_model(
            str(Path(model_path).resolve()),
            str(Path(labels_path).resolve()),
        )
        prediction = predict_ripeness(image_bgr, model, labels)
        return {
            "available": True,
            "model_path": str(model_path),
            **prediction,
        }
    except Exception as exc:
        return {
            "available": False,
            "error": f"Ripeness classification could not run: {exc}",
        }
