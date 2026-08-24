"""ROI drawing helpers for GrabCut Mode B."""

from __future__ import annotations

import hashlib

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

def roi_percent_from_canvas(json_data: dict | None, width: int, height: int):
    """Return newest drawn rectangle as left/top/right/bottom percentages."""
    if not json_data:
        return None
    rects = [o for o in json_data.get("objects", []) if o.get("type") == "rect"]
    if not rects:
        return None
    r = rects[-1]
    left = float(r.get("left", 0.0))
    top = float(r.get("top", 0.0))
    rw = float(r.get("width", 0.0)) * float(r.get("scaleX", 1.0))
    rh = float(r.get("height", 0.0)) * float(r.get("scaleY", 1.0))
    x1 = max(0.0, min(left, width - 1.0))
    y1 = max(0.0, min(top, height - 1.0))
    x2 = max(x1 + 1.0, min(left + rw, float(width)))
    y2 = max(y1 + 1.0, min(top + rh, float(height)))
    return (100*x1/width, 100*y1/height, 100*x2/width, 100*y2/height)


def render_grabcut_roi(image: np.ndarray):
    """Render the original ROI canvas and return rectangle percentages."""
    st.subheader("Draw ROI for Mode B")
    st.caption(
        "Keep the complete banana inside the rectangle with a small background margin. "
        "Do not make the ROI unnecessarily large."
    )
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bg = Image.fromarray(rgb)
    oh, ow = image.shape[:2]
    canvas_w = min(760, max(320, ow))
    canvas_h = max(180, int(round(canvas_w * oh / ow)))
    image_key = hashlib.md5(image.tobytes()).hexdigest()[:12]
    canvas = st_canvas(
        fill_color="rgba(0,255,0,0.08)",
        stroke_width=3,
        stroke_color="#00ff00",
        background_image=bg,
        update_streamlit=True,
        height=canvas_h,
        width=canvas_w,
        drawing_mode="rect",
        display_toolbar=True,
        key=f"banana_roi_{image_key}",
    )
    grabcut_roi_percent = roi_percent_from_canvas(canvas.json_data, canvas_w, canvas_h)
    if grabcut_roi_percent is None:
        st.info("Draw one rectangle ROI around the banana to run Mode B.")
        st.stop()
    return grabcut_roi_percent
