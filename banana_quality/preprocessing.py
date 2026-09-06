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
    """Shared resize/enhancement/blur/colour conversion for image analysis."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Input image is empty")

    # Resize large images while preserving their original aspect ratio.
    working, scale = resize_keep_aspect(image_bgr, config.max_dimension)

    # CLAHE contrast enhancement.
    working_lab = cv2.cvtColor(working, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(working_lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )
    enhanced_lightness = clahe.apply(lightness)

    enhanced_lab = cv2.merge(
        (enhanced_lightness, channel_a, channel_b)
    )
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # 3. Gaussian filtering reduces small image noise before clustering.
    blur_kernel = odd_kernel(config.gaussian_kernel)
    blurred = cv2.GaussianBlur(
        enhanced,
        (blur_kernel, blur_kernel),
        0,
    )

    # 4. Lab is useful for colour clustering; HSV is useful for broad
    # banana-colour rules used when selecting K-means clusters.
    lab_u8 = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    return {
        "working_image": working,
        "clahe_enhanced": enhanced,
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
