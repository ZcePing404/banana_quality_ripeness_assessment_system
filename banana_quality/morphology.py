from __future__ import annotations

import cv2
import numpy as np

from .image_utils import odd_kernel

def _morphology(mask: np.ndarray, operation: int, kernel_size: int) -> np.ndarray:
    """Apply one morphology operation. 1x1 is intentionally treated as disabled."""
    kernel_size = odd_kernel(kernel_size)
    if kernel_size <= 1:
        return mask.copy()

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    return cv2.morphologyEx(mask, operation, kernel, iterations=1)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep the largest connected foreground region (project scope = one banana)."""
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return np.zeros_like(mask)

    areas = stats[1:, cv2.CC_STAT_AREA]
    best_label = 1 + int(np.argmax(areas))
    return np.where(labels == best_label, 255, 0).astype(np.uint8)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill holes inside the banana, including when the banana touches an image edge."""
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    height, width = binary.shape

    inverse = cv2.bitwise_not(binary)
    padded = cv2.copyMakeBorder(inverse, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
    flood = padded.copy()
    flood_mask = np.zeros((height + 4, width + 4), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 0)
    holes = flood[1:-1, 1:-1]
    return cv2.bitwise_or(binary, holes)


def _remove_tiny_components(mask: np.ndarray, minimum_area: int) -> np.ndarray:
    """Remove tiny isolated blemish regions that are likely to be noise."""
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(mask)

    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= int(minimum_area):
            cleaned[labels == label] = 255
    return cleaned

