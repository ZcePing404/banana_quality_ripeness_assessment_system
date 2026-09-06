from __future__ import annotations

import cv2
import numpy as np

def odd_kernel(value: int) -> int:
    # Return an odd kernel size >= 1. A 1x1 kernel means 'no morphology'.
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def resize_keep_aspect(image_bgr: np.ndarray, max_dimension: int) -> tuple[np.ndarray, float]:
    # Resize only when necessary and preserve the image aspect ratio.
    height, width = image_bgr.shape[:2]
    scale = min(1.0, float(max_dimension) / max(height, width))
    if scale >= 0.9999:
        return image_bgr.copy(), 1.0

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, scale

