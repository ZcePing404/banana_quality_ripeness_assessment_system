"""Common image preparation used by segmentation and blemish analysis."""

from __future__ import annotations

import cv2
import numpy as np

from .config import ProcessingConfig
from .image_utils import odd_kernel, resize_keep_aspect

def _common_preprocess(
    image_bgr: np.ndarray,
    config: ProcessingConfig,
) -> dict[str, object]:
    """Shared resize/blur/colour conversion used by both segmentation modes."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Input image is empty")

    working, scale = resize_keep_aspect(image_bgr, config.max_dimension)
    blur_kernel = odd_kernel(config.gaussian_kernel)
    blurred = cv2.GaussianBlur(working, (blur_kernel, blur_kernel), 0)

    # Lab is useful for colour clustering; HSV is useful for broad banana-colour rules.
    lab_u8 = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    return {
        "working_image": working,
        "blurred": blurred,
        "lab": lab_u8,
        "hsv": hsv,
        "scale": float(scale),
    }


def prepare_blemish_image(
    image_bgr: np.ndarray,
    config: ProcessingConfig | None = None,
) -> dict[str, object]:
    """Prepare an image for blemish analysis without running banana segmentation.

    This is useful during evaluation when a ground-truth banana mask is supplied
    so the blemish module can be measured independently from banana segmentation.
    """
    config = config or ProcessingConfig()
    return _common_preprocess(image_bgr, config)

