"""High-level orchestration for banana segmentation and blemish analysis.

The individual image-processing techniques are implemented in dedicated modules.
This file intentionally stays small and coordinates those modules while preserving
the original public imports used by the application and evaluation script.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .blemish.detection import blemish_mask_from_prepared
from .config import ProcessingConfig
from .image_utils import odd_kernel
from .preprocessing import prepare_blemish_image
from .segmentation.grabcut import prepare_grabcut_image
from .segmentation.kmeans import prepare_image
from .segmentation.mask_refinement import banana_mask_from_prepared
from .surface_quality import _resize_mask_to_original, blemish_percentage, quality_grade

def analyse_image(
    image_bgr: np.ndarray,
    *,
    segmentation_mode: str = "mode_a",
    kmeans_k: int | None = None,
    grabcut_roi_percent: tuple[float, float, float, float] | None = None,
    grabcut_rect_percent: tuple[float, float, float, float] | None = None,
    banana_opening_kernel: int = 3,
    banana_closing_kernel: int = 5,
    blemish_opening_kernel: int = 3,
    blemish_closing_kernel: int = 5,
    config: ProcessingConfig | None = None,
) -> dict[str, object]:
    """Run independent Mode A (K-means) or Mode B (ROI + GrabCut)."""
    config = config or ProcessingConfig()
    if kmeans_k is not None:
        config = replace(config, kmeans_k=int(kmeans_k))

    if grabcut_roi_percent is None:
        grabcut_roi_percent = (
            grabcut_rect_percent
            if grabcut_rect_percent is not None
            else (5.0, 5.0, 95.0, 95.0)
        )

    mode = str(segmentation_mode).strip().lower()

    if mode in {"mode_a", "a", "kmeans", "automatic", "automatic k-means"}:
        mode = "mode_a"
        prepared = prepare_image(image_bgr, config)
        # K-means masks may contain small noise/gaps, so Mode A keeps adjustable morphology.
        banana_open = banana_opening_kernel
        banana_close = banana_closing_kernel

    elif mode in {"mode_b", "b", "grabcut", "assisted", "assisted grabcut"}:
        mode = "mode_b"
        prepared = prepare_grabcut_image(
            image_bgr,
            roi_percent=grabcut_roi_percent,
            config=config,
        )
        # IMPORTANT: Mode B is GrabCut + ROI only. No K-means and no banana morphology.
        banana_open = 1
        banana_close = 1

    else:
        raise ValueError("segmentation_mode must be Mode A/K-means or Mode B/ROI+GrabCut")

    banana_steps = banana_mask_from_prepared(
        prepared,
        opening_kernel_size=banana_open,
        closing_kernel_size=banana_close,
    )

    blemish_steps = blemish_mask_from_prepared(
        prepared,
        np.asarray(banana_steps["banana_mask"], dtype=np.uint8),
        opening_kernel_size=blemish_opening_kernel,
        closing_kernel_size=blemish_closing_kernel,
        config=config,
    )

    original_shape = image_bgr.shape[:2]
    banana_original = _resize_mask_to_original(
        np.asarray(banana_steps["banana_mask"], dtype=np.uint8),
        original_shape,
    )
    blemish_original = _resize_mask_to_original(
        np.asarray(blemish_steps["blemish_mask"], dtype=np.uint8),
        original_shape,
    )

    percentage = blemish_percentage(banana_original, blemish_original)

    return {
        "segmentation_mode": mode,
        "prepared": prepared,
        "banana_steps": banana_steps,
        "blemish_steps": blemish_steps,
        "banana_mask": banana_original,
        "blemish_mask": blemish_original,
        "blemish_percentage": percentage,
        "quality_grade": quality_grade(percentage),
        "kmeans_k": int(config.kmeans_k),
        "grabcut_roi_percent": tuple(float(v) for v in grabcut_roi_percent),
        "grabcut_rect_percent": tuple(float(v) for v in grabcut_roi_percent),
        "banana_opening_kernel": odd_kernel(banana_open),
        "banana_closing_kernel": odd_kernel(banana_close),
        "blemish_opening_kernel": odd_kernel(blemish_opening_kernel),
        "blemish_closing_kernel": odd_kernel(blemish_closing_kernel),
    }

__all__ = [
    "ProcessingConfig",
    "analyse_image",
    "prepare_blemish_image",
    "prepare_image",
    "prepare_grabcut_image",
    "banana_mask_from_prepared",
    "blemish_mask_from_prepared",
    "blemish_percentage",
    "quality_grade",
]
