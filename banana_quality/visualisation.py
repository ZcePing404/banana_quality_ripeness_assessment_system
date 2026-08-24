"""Small visualisation helpers for banana quality output."""

from __future__ import annotations

import cv2
import numpy as np


def overlay_result(
    image_bgr: np.ndarray,
    banana_mask: np.ndarray,
    blemish_mask: np.ndarray,
) -> np.ndarray:
    """Draw green banana boundary and translucent red blemish overlay."""
    output = image_bgr.copy()

    banana = np.where(banana_mask > 0, 255, 0).astype(np.uint8)
    blemish = blemish_mask > 0

    if np.any(blemish):
        red_layer = output.copy()
        red_layer[blemish] = (0, 0, 255)
        output = cv2.addWeighted(output, 0.65, red_layer, 0.35, 0)

    contours, _ = cv2.findContours(
        banana,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
    return output
