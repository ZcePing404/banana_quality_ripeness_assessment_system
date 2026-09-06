"""Flask bridge for baseline and segmented Teachable Machine classifiers."""

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

MODEL_ROOT = (
    PROJECT_ROOT
    / "machine_learning"
    / "ripeness_classification"
    / "models"
)

BASELINE_MODEL_DIR = MODEL_ROOT / "baseline"
SEGMENTED_MODEL_DIR = MODEL_ROOT / "segmented"


@lru_cache(maxsize=4)
def _load_cached_model(
    model_path: str,
    labels_path: str,
):
    """Load each Teachable Machine model only once per Flask process."""
    return load_ripeness_model(
        model_path,
        labels_path,
    )


def _model_files(method: str) -> tuple[Path, Path]:
    """Resolve the model and labels belonging to one classification method."""

    if method == "baseline":
        model_dir = BASELINE_MODEL_DIR
    elif method == "segmented":
        model_dir = SEGMENTED_MODEL_DIR
    else:
        raise ValueError(
            "Ripeness method must be either 'baseline' or 'segmented'."
        )

    return (
        model_dir / "keras_model.h5",
        model_dir / "labels.txt",
    )


def classify_ripeness(
    image_bgr: np.ndarray,
    banana_mask: np.ndarray | None = None,
    *,
    method: str = "baseline",
) -> dict[str, object]:
    """Classify one banana using the selected trained model.

    BASELINE pipeline
    -----------------
        original/cropped image
            -> Teachable Machine preprocessing
            -> baseline model

    The baseline branch does NOT call apply_banana_mask() and therefore does
    not perform background removal for ripeness classification.

    SEGMENTED pipeline
    ------------------
        original/cropped image
            -> final shared K-means banana mask
            -> white background outside the banana
            -> Teachable Machine preprocessing
            -> segmented model

    The segmented branch reuses the final banana mask already produced by
    banana_quality.pipeline.analyse_image(); it does not run a second K-means
    segmentation.
    """

    method = str(method).strip().lower()

    if method not in {"baseline", "segmented"}:
        return {
            "available": False,
            "method": method,
            "error": (
                "Unknown ripeness method. Choose 'baseline' or 'segmented'."
            ),
        }

    if image_bgr is None or np.asarray(image_bgr).size == 0:
        return {
            "available": False,
            "method": method,
            "error": "Ripeness classification received an empty image.",
        }

    try:
        model_path, labels_path = _model_files(method)

        if not model_path.exists():
            return {
                "available": False,
                "method": method,
                "error": (
                    f"{method.title()} ripeness model was not found at: "
                    f"{model_path}"
                ),
            }

        if not labels_path.exists():
            return {
                "available": False,
                "method": method,
                "error": (
                    f"{method.title()} model labels were not found at: "
                    f"{labels_path}"
                ),
            }

        if method == "baseline":
            # IMPORTANT: no banana-mask application and no background removal.
            classifier_input = np.asarray(image_bgr).copy()
            shared_mask = None

            method_label = "Baseline model"
            input_caption = "Original / cropped classifier input"

        else:
            if banana_mask is None or np.asarray(banana_mask).size == 0:
                return {
                    "available": False,
                    "method": method,
                    "error": (
                        "Segmented classification requires the final banana mask."
                    ),
                }

            # Reuse the exact final K-means mask from the main surface pipeline.
            classifier_input, shared_mask = apply_banana_mask(
                image_bgr,
                banana_mask,
            )

            method_label = "Segmented model"
            input_caption = "Background-removed classifier input"

        model, labels = _load_cached_model(
            str(model_path.resolve()),
            str(labels_path.resolve()),
        )

        prediction = predict_ripeness(
            classifier_input,
            model,
            labels,
        )

        return {
            "available": True,
            "method": method,
            "method_label": method_label,
            "model_path": str(model_path),
            "classifier_input": classifier_input,
            "banana_mask": shared_mask,
            "input_caption": input_caption,

            # Backward-compatible alias for older Flask UI code.
            "segmented_image": (
                classifier_input
                if method == "segmented"
                else None
            ),

            **prediction,
        }

    except Exception as exc:
        return {
            "available": False,
            "method": method,
            "error": (
                "Ripeness classification could not run: "
                f"{exc}"
            ),
        }
