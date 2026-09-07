from __future__ import annotations

import cv2
import numpy as np


def _as_line(points: list[list[float]] | np.ndarray) -> np.ndarray:
    """Return exactly two points as float32 [[x1, y1], [x2, y2]]."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if pts.shape != (2, 2):
        raise ValueError("Calibration requires exactly two reference points.")

    length_px = float(np.linalg.norm(pts[1] - pts[0]))
    if not np.isfinite(length_px) or length_px < 5.0:
        raise ValueError("The selected calibration line is too short or invalid.")
    return pts


def calculate_pixels_per_cm_from_line(
    reference_points: list[list[float]] | np.ndarray,
    reference_length_cm: float,
) -> float:
    """Convert a known reference line into a pixel/cm scale."""
    reference_length_cm = float(reference_length_cm)
    if not np.isfinite(reference_length_cm) or reference_length_cm <= 0:
        raise ValueError("Reference length must be greater than 0 cm.")

    pts = _as_line(reference_points)
    pixel_length = float(np.linalg.norm(pts[1] - pts[0]))
    pixels_per_cm = pixel_length / reference_length_cm
    if not np.isfinite(pixels_per_cm) or pixels_per_cm <= 0:
        raise ValueError("Could not compute a valid pixels-per-cm scale.")
    return pixels_per_cm


def measure_banana_from_mask(
    banana_mask: np.ndarray,
    pixels_per_cm: float,
) -> dict[str, object]:
    """Estimate banana size from the largest contour in the banana mask."""
    mask = np.asarray(banana_mask)
    if mask.size == 0:
        raise ValueError("Banana mask is empty.")

    binary = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No banana contour was found for measurement.")

    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    (center_x, center_y), (side_a, side_b), angle = rect
    length_px = float(max(side_a, side_b))
    width_px = float(min(side_a, side_b))

    pixels_per_cm = float(pixels_per_cm)
    if not np.isfinite(pixels_per_cm) or pixels_per_cm <= 0:
        raise ValueError("pixels_per_cm must be greater than 0.")

    box = cv2.boxPoints(rect).astype(np.int32)
    return {
        "contour": largest,
        "box": box,
        "center": (float(center_x), float(center_y)),
        "angle": float(angle),
        "length_px": length_px,
        "width_px": width_px,
        "length_cm": length_px / pixels_per_cm,
        "width_cm": width_px / pixels_per_cm,
        "pixels_per_cm": pixels_per_cm,
    }


def draw_measurement_box(
    image_bgr: np.ndarray,
    measurement: dict[str, object],
) -> np.ndarray:
    """Draw a rotated measurement box and label on an output image."""
    output = np.asarray(image_bgr).copy()
    box = np.asarray(measurement.get("box"), dtype=np.int32).reshape(-1, 1, 2)
    cv2.drawContours(output, [box], 0, (255, 140, 0), 2)

    label = (
        f"Estimated size: {float(measurement['length_cm']):.1f} x "
        f"{float(measurement['width_cm']):.1f} cm"
    )

    x = int(np.min(box[:, 0, 0]))
    y = int(np.min(box[:, 0, 1])) - 10
    y = max(24, y)

    (label_w, label_h), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        2,
    )
    pad = 8
    cv2.rectangle(
        output,
        (x - pad, y - label_h - pad),
        (x + label_w + pad, y + baseline + pad),
        (255, 255, 255),
        -1,
    )
    cv2.rectangle(
        output,
        (x - pad, y - label_h - pad),
        (x + label_w + pad, y + baseline + pad),
        (255, 140, 0),
        2,
    )
    cv2.putText(
        output,
        label,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (34, 34, 34),
        2,
        cv2.LINE_AA,
    )
    return output
