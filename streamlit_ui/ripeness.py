"""Streamlit bridge for the segmented Teachable Machine ripeness classifier."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st

from machine_learning.ripeness_classification.predictor import (
    load_ripeness_model,
    predict_ripeness,
)

from machine_learning.ripeness_classification.segmentation import (
    segment_banana,
)


# ============================================================
# MODEL PATH
# ============================================================

# ripeness.py is inside:
# project_root/streamlit_ui/ripeness.py
#
# Therefore parents[1] gives the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEGMENTED_MODEL_DIR = (
    PROJECT_ROOT
    / "machine_learning"
    / "ripeness_classification"
    / "models"
    / "segmented"
)

MODEL_PATH = (
    SEGMENTED_MODEL_DIR
    / "keras_model.h5"
)

LABELS_PATH = (
    SEGMENTED_MODEL_DIR
    / "labels.txt"
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner=False)
def _load_cached_model(
    model_path: str,
    labels_path: str,
):
    """
    Load the Teachable Machine model only once
    for the current Streamlit process.
    """

    return load_ripeness_model(
        model_path,
        labels_path,
    )


# ============================================================
# STREAMLIT RIPENESS CLASSIFICATION
# ============================================================

def classify_ripeness_for_ui(
    image_bgr: np.ndarray,
) -> dict[str, object]:
    """
    Classify one banana image.

    Pipeline:

        Original image
            ↓
        Banana segmentation
            ↓
        Background removal
            ↓
        Segmented Teachable Machine model
            ↓
        Unripe / Ripe / Overripe

    The same segment_banana() function is used to create
    the segmented train, validation and test datasets.
    """

    # --------------------------------------------------------
    # 1. Check that the segmented model exists
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        return {
            "available": False,
            "error": (
                "Segmented ripeness model was not found at: "
                f"{MODEL_PATH}"
            ),
        }

    if not LABELS_PATH.exists():

        return {
            "available": False,
            "error": (
                "Segmented model labels were not found at: "
                f"{LABELS_PATH}"
            ),
        }

    # --------------------------------------------------------
    # 2. Check input image
    # --------------------------------------------------------

    if (
        image_bgr is None
        or image_bgr.size == 0
    ):

        return {
            "available": False,
            "error": (
                "Ripeness classification received "
                "an empty image."
            ),
        }

    try:

        # ====================================================
        # 3. SEGMENT THE BANANA
        # ====================================================

        # This is the EXACT SAME function used by
        # segment_dataset.py to generate:
        #
        # data/segmented/train
        # data/segmented/valid
        # data/segmented/test

        segmented_image, banana_mask = (
            segment_banana(image_bgr)
        )

        if (
            segmented_image is None
            or segmented_image.size == 0
        ):
            raise ValueError(
                "Banana segmentation produced "
                "an empty image."
            )

        if (
            banana_mask is None
            or banana_mask.size == 0
        ):
            raise ValueError(
                "Banana segmentation produced "
                "an empty mask."
            )

        # ====================================================
        # 4. LOAD SEGMENTED MODEL
        # ====================================================

        model, labels = _load_cached_model(
            str(MODEL_PATH.resolve()),
            str(LABELS_PATH.resolve()),
        )

        # ====================================================
        # 5. CLASSIFY THE SEGMENTED IMAGE
        # ====================================================

        # IMPORTANT:
        #
        # Previously this was:
        #
        # predict_ripeness(image_bgr, ...)
        #
        # Now the classifier receives the segmented image
        # because that is what this model was trained on.

        prediction = predict_ripeness(
            segmented_image,
            model,
            labels,
        )

        # ====================================================
        # 6. RETURN RESULT TO STREAMLIT
        # ====================================================

        return {
            "available": True,

            "model_path":
                str(MODEL_PATH),

            # Keep these temporarily.
            # They are useful for checking that Streamlit
            # uses the correct segmentation.
            "segmented_image":
                segmented_image,

            "banana_mask":
                banana_mask,

            **prediction,
        }

    except Exception as exc:

        return {
            "available": False,
            "error": (
                "Ripeness classification could not run: "
                f"{exc}"
            ),
        }