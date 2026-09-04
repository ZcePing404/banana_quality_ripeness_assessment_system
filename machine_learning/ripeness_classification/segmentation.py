"""Banana segmentation helpers for ripeness-classification preprocessing.

This module now reuses the project's shared K-means banana-mask algorithm.
GrabCut and the old fixed ``K = 4`` preprocessing have been removed.

For Streamlit, the banana mask is created once by the surface-quality pipeline
using the current sidebar K value.  That exact mask is then applied here so
both blemish analysis and ripeness classification use the same banana region.
"""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from banana_quality.config import ProcessingConfig
from banana_quality.segmentation.kmeans import prepare_image
from banana_quality.segmentation.mask_refinement import banana_mask_from_prepared
from banana_quality.surface_quality import _resize_mask_to_original


BACKGROUND_VALUE = 255


def _normalise_mask(mask: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """Return a single-channel uint8 mask aligned with the source image."""
    if mask is None or np.asarray(mask).size == 0:
        raise ValueError("Banana mask is empty.")

    mask_array = np.asarray(mask)

    if mask_array.ndim == 3:
        mask_array = cv2.cvtColor(mask_array.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    elif mask_array.ndim != 2:
        raise ValueError("Banana mask must be a 2-D binary mask.")

    target_height, target_width = image_shape
    if mask_array.shape != (target_height, target_width):
        mask_array = cv2.resize(
            mask_array,
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST,
        )

    return np.where(mask_array > 0, 255, 0).astype(np.uint8)


def remove_background(
    image_bgr: np.ndarray,
    banana_mask: np.ndarray,
    *,
    background_value: int = BACKGROUND_VALUE,
) -> np.ndarray:
    """Keep banana pixels and replace all pixels outside the mask with white."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Cannot apply a banana mask to an empty image.")

    mask = _normalise_mask(banana_mask, image_bgr.shape[:2])
    output = np.full_like(image_bgr, int(background_value))
    output[mask > 0] = image_bgr[mask > 0]
    return output


def apply_banana_mask(
    image_bgr: np.ndarray,
    banana_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply an already-computed shared banana mask for ripeness input.

    This is the function used by Streamlit.  It does not run another K-means
    pass and it does not use GrabCut.  Therefore the ripeness classifier sees
    the same banana boundary used by blemish analysis.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Cannot prepare an empty image for ripeness classification.")

    mask = _normalise_mask(banana_mask, image_bgr.shape[:2])
    segmented = remove_background(image_bgr, mask)
    return segmented, mask


def segment_banana(
    image_bgr: np.ndarray,
    *,
    kmeans_k: int | None = None,
    banana_score_threshold: float | None = None,
    opening_kernel_size: int | None = None,
    closing_kernel_size: int | None = None,
    config: ProcessingConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a banana mask with the shared K-means algorithm.

    ``kmeans_k`` is configurable rather than fixed at 4.  If omitted, the
    project's tuned ``ProcessingConfig`` default is used.  This helper is
    useful for regenerating segmented training/validation/test images.  The
    interactive Streamlit application instead passes its already-computed mask
    to :func:`apply_banana_mask` so the mask is generated only once.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Cannot segment an empty image.")

    cfg = config or ProcessingConfig()

    updates: dict[str, object] = {}
    if kmeans_k is not None:
        updates["kmeans_k"] = int(kmeans_k)
    if banana_score_threshold is not None:
        updates["banana_cluster_score_threshold"] = float(banana_score_threshold)
    if updates:
        cfg = replace(cfg, **updates)

    opening = (
        int(opening_kernel_size)
        if opening_kernel_size is not None
        else int(cfg.banana_opening_kernel)
    )
    closing = (
        int(closing_kernel_size)
        if closing_kernel_size is not None
        else int(cfg.banana_closing_kernel)
    )

    prepared = prepare_image(image_bgr, cfg)
    banana_steps = banana_mask_from_prepared(
        prepared,
        opening_kernel_size=opening,
        closing_kernel_size=closing,
    )

    mask = _resize_mask_to_original(
        np.asarray(banana_steps["banana_mask"], dtype=np.uint8),
        image_bgr.shape[:2],
    )
    mask = _normalise_mask(mask, image_bgr.shape[:2])
    segmented = remove_background(image_bgr, mask)

    return segmented, mask


__all__ = [
    "BACKGROUND_VALUE",
    "apply_banana_mask",
    "remove_background",
    "segment_banana",
]
