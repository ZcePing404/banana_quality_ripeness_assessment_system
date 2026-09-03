"""Image upload decoding helpers."""

from __future__ import annotations

import cv2
import numpy as np

def decode_image(file_value) -> np.ndarray | None:
    """Convert an uploaded image into an OpenCV BGR image."""
    if file_value is None:
        return None
    raw = np.frombuffer(file_value.getvalue(), dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)

