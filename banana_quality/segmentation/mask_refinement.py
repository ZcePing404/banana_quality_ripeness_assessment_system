"""Shared banana-mask refinement applied after segmentation."""

from __future__ import annotations

import cv2
import numpy as np

from ..image_utils import odd_kernel
from ..morphology import _fill_holes, _largest_component, _morphology

def banana_mask_from_prepared(
    prepared: dict[str, object],
    opening_kernel_size: int = 3,
    closing_kernel_size: int = 5,
) -> dict[str, np.ndarray | int]:
    """Convert the current raw banana candidate into one clean banana mask."""
    raw = np.asarray(prepared["raw_banana_candidate"], dtype=np.uint8)
    opening_kernel = odd_kernel(opening_kernel_size)
    closing_kernel = odd_kernel(closing_kernel_size)

    opened = _morphology(raw, cv2.MORPH_OPEN, opening_kernel)
    closed = _morphology(opened, cv2.MORPH_CLOSE, closing_kernel)

    # The project scope is one banana, therefore only one dominant foreground
    # object is retained before filling holes.
    largest = _largest_component(closed)
    final_mask = _fill_holes(largest)

    return {
        "banana_opened": opened,
        "banana_closed": closed,
        "banana_largest_component": largest,
        "banana_mask": final_mask,
        "banana_opening_kernel": opening_kernel,
        "banana_closing_kernel": closing_kernel,
    }

