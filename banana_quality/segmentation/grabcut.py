"""ROI-guided GrabCut banana segmentation preparation (Mode B)."""

from __future__ import annotations

import cv2
import numpy as np

from ..config import ProcessingConfig
from ..preprocessing import _common_preprocess

def _rectangle_from_percent(
    width: int,
    height: int,
    rect_percent: tuple[float, float, float, float],
) -> tuple[int, int, int, int]:
    """Convert (left, top, right, bottom) percentages into a valid GrabCut rectangle."""
    left, top, right, bottom = [float(v) for v in rect_percent]

    left = min(max(left, 0.0), 98.0)
    top = min(max(top, 0.0), 98.0)
    right = min(max(right, left + 1.0), 100.0)
    bottom = min(max(bottom, top + 1.0), 100.0)

    x1 = int(round(width * left / 100.0))
    y1 = int(round(height * top / 100.0))
    x2 = int(round(width * right / 100.0))
    y2 = int(round(height * bottom / 100.0))

    # Keep at least a one-pixel outer margin. GrabCut needs some pixels outside
    # the rectangle to act as definite background guidance.
    x1 = min(max(x1, 0), max(0, width - 3))
    y1 = min(max(y1, 0), max(0, height - 3))
    x2 = min(max(x2, x1 + 2), width - 1)
    y2 = min(max(y2, y1 + 2), height - 1)

    return (x1, y1, x2 - x1, y2 - y1)


def _colourise_grabcut_labels(label_map: np.ndarray) -> np.ndarray:
    """Make GrabCut's four internal labels easy to understand visually.

    BGR colours used here:
    - black  = definite background (GC_BGD)
    - blue   = probable background (GC_PR_BGD)
    - yellow = probable foreground (GC_PR_FGD)
    - green  = definite foreground (GC_FGD)

    Rectangle-initialised GrabCut often contains mostly definite background,
    probable background and probable foreground. A definite-foreground label
    is more common when explicit foreground seeds are supplied.
    """
    colour = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    colour[label_map == cv2.GC_BGD] = (0, 0, 0)
    colour[label_map == cv2.GC_PR_BGD] = (255, 0, 0)
    colour[label_map == cv2.GC_PR_FGD] = (0, 255, 255)
    colour[label_map == cv2.GC_FGD] = (0, 255, 0)
    return colour


def prepare_grabcut_image(
    image_bgr: np.ndarray,
    roi_percent: tuple[float, float, float, float] = (5.0, 5.0, 95.0, 95.0),
    config: ProcessingConfig | None = None,
) -> dict[str, object]:
    """Prepare ASSISTED GrabCut banana segmentation.

    The user rectangle should contain the banana and ideally leave some background
    outside it. Pixels outside the rectangle are treated as definite background
    guidance. Pixels inside are allowed to be foreground or background.
    """
    config = config or ProcessingConfig()
    prepared = _common_preprocess(image_bgr, config)

    working = np.asarray(prepared["working_image"], dtype=np.uint8)
    blurred = np.asarray(prepared["blurred"], dtype=np.uint8)
    height, width = working.shape[:2]

    rect = _rectangle_from_percent(width, height, roi_percent)

    # Visualise the information that the rectangle gives GrabCut BEFORE it runs:
    # outside rectangle = definite background, inside = probable foreground.
    # This is only an explanatory image; OpenCV creates the same kind of
    # initialisation internally when GC_INIT_WITH_RECT is used.
    x, y, w, h = rect
    initial_gc_mask = np.full((height, width), cv2.GC_BGD, dtype=np.uint8)
    initial_gc_mask[y : y + h, x : x + w] = cv2.GC_PR_FGD
    initial_label_colour = _colourise_grabcut_labels(initial_gc_mask)

    # GrabCut internally learns separate colour models for foreground and
    # background, then uses graph-cut optimisation so neighbouring pixels also
    # influence the final separation. The rectangle is only a starting hint.
    gc_mask = np.zeros((height, width), dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    cv2.grabCut(
        blurred,
        gc_mask,
        rect,
        bgd_model,
        fgd_model,
        int(config.grabcut_iterations),
        cv2.GC_INIT_WITH_RECT,
    )

    raw_banana = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)

    # Draw the user rectangle only for visual explanation in the GUI/notebook.
    preview = working.copy()
    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # GrabCut produces four labels. A colour image is much easier to understand
    # than the previous grey 0/85/170/255 display.
    final_label_colour = _colourise_grabcut_labels(gc_mask)

    # Show exactly what GrabCut itself considers foreground BEFORE morphology,
    # largest-component selection and hole filling. This makes it easy to tell
    # whether a mistake comes from GrabCut or from the later cleanup stages.
    foreground_preview = np.zeros_like(working)
    foreground_preview[raw_banana > 0] = working[raw_banana > 0]

    prepared.update(
        {
            "segmentation_mode": "grabcut",
            "grabcut_rectangle": rect,
            "grabcut_roi_percent": tuple(float(v) for v in roi_percent),
            "grabcut_rectangle_percent": tuple(float(v) for v in roi_percent),
            "grabcut_rectangle_preview": preview,
            "grabcut_initial_label_colour": initial_label_colour,
            "grabcut_final_label_colour": final_label_colour,
            "grabcut_label_map": gc_mask.copy(),
            "grabcut_foreground_preview": foreground_preview,
            "raw_banana_candidate": raw_banana,
        }
    )
    return prepared

