"""Blemish-area calculation and surface-quality grading."""

from __future__ import annotations

import cv2
import numpy as np

def _resize_mask_to_original(mask: np.ndarray, original_shape: tuple[int, int]) -> np.ndarray:
    original_height, original_width = original_shape
    if mask.shape == (original_height, original_width):
        return mask.copy()
    return cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)


def blemish_percentage(banana_mask: np.ndarray, blemish_mask: np.ndarray) -> float:
    """Percentage of the detected banana area classified as blemish."""
    banana = banana_mask > 0
    banana_pixels = int(banana.sum())
    if banana_pixels == 0:
        return 0.0
    blemish_pixels = int(np.logical_and(blemish_mask > 0, banana).sum())
    return float(blemish_pixels / banana_pixels * 100.0)


def quality_grade(percentage: float) -> str:
    """Simple surface-quality bands retained from the earlier project."""
    if percentage <= 1.0:
        return "Extra"
    if percentage <= 5.0:
        return "Class I"
    if percentage <= 10.0:
        return "Class II"
    return "Reject"

