from __future__ import annotations

from dataclasses import replace

import numpy as np

from .blemish.detection import blemish_mask_from_prepared
from .config import ProcessingConfig
from .image_utils import odd_kernel
from .preprocessing import prepare_blemish_image
from .segmentation.kmeans import prepare_image
from .segmentation.mask_refinement import banana_mask_from_prepared
from .surface_quality import _resize_mask_to_original, blemish_percentage, quality_grade


def analyse_image(
    image_bgr: np.ndarray,
    *,
    segmentation_mode: str = "mode_a",
    kmeans_k: int | None = None,
    banana_opening_kernel: int = 3,
    banana_closing_kernel: int = 5,
    blemish_opening_kernel: int = 3,
    blemish_closing_kernel: int = 5,
    config: ProcessingConfig | None = None,
) -> dict[str, object]:
    # Run automatic K-means banana segmentation followed by blemish analysis.
    mode = str(segmentation_mode).strip().lower()
    if mode not in {"mode_a", "a", "kmeans", "automatic", "automatic k-means"}:
        raise ValueError("Only automatic K-means banana segmentation is supported.")

    config = config or ProcessingConfig()
    if kmeans_k is not None:
        config = replace(config, kmeans_k=int(kmeans_k))

    prepared = prepare_image(image_bgr, config)

    banana_steps = banana_mask_from_prepared(
        prepared,
        opening_kernel_size=banana_opening_kernel,
        closing_kernel_size=banana_closing_kernel,
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
        "segmentation_mode": "mode_a",
        "prepared": prepared,
        "banana_steps": banana_steps,
        "blemish_steps": blemish_steps,
        "banana_mask": banana_original,
        "blemish_mask": blemish_original,
        "blemish_percentage": percentage,
        "quality_grade": quality_grade(percentage),
        "kmeans_k": int(config.kmeans_k),
        "banana_opening_kernel": odd_kernel(banana_opening_kernel),
        "banana_closing_kernel": odd_kernel(banana_closing_kernel),
        "blemish_opening_kernel": odd_kernel(blemish_opening_kernel),
        "blemish_closing_kernel": odd_kernel(blemish_closing_kernel),
    }


__all__ = [
    "ProcessingConfig",
    "analyse_image",
    "prepare_blemish_image",
    "prepare_image",
    "banana_mask_from_prepared",
    "blemish_mask_from_prepared",
    "blemish_percentage",
    "quality_grade",
]
