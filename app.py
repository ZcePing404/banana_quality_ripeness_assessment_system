"""Streamlit entry point for banana mask and blemish detection."""

from __future__ import annotations

from dataclasses import replace

import streamlit as st

# Streamlit requires page configuration before the rest of the UI is created.
st.set_page_config(page_title="Banana Blemish Detection", layout="wide")

from banana_quality.config import ProcessingConfig
from banana_quality.pipeline import analyse_image
from streamlit_ui.controls import render_sidebar
from streamlit_ui.input_section import render_image_input
from streamlit_ui.results import render_results
from streamlit_ui.roi import render_grabcut_roi


st.title("Banana Mask and Blemish Detection")
st.caption(
    "Mode A uses K-means for automatic banana segmentation. Mode B uses a "
    "user-drawn ROI with GrabCut for banana segmentation. Both modes share the "
    "second-stage blemish detector."
)

settings = render_sidebar()
is_mode_a = bool(settings["is_mode_a"])

image, input_source = render_image_input()

if image is None:
    if input_source == "Upload image":
        st.info("Upload an image to start.")
    else:
        st.info(
            "The live preview can continue above. Press **Capture current frame** "
            "when the banana is positioned correctly."
        )
    st.stop()

grabcut_roi_percent = None
if not is_mode_a:
    grabcut_roi_percent = render_grabcut_roi(image)

config = replace(
    ProcessingConfig(),
    banana_cluster_score_threshold=float(settings["banana_score_threshold"]),
    blemish_kmeans_k=int(settings["blemish_k"]),
    blemish_cluster_chroma_threshold=float(settings["blemish_chroma_threshold"]),
    darkness_threshold=float(settings["darkness_threshold"]),
)

result = analyse_image(
    image,
    segmentation_mode="mode_a" if is_mode_a else "mode_b",
    kmeans_k=int(settings["kmeans_k"]),
    grabcut_roi_percent=grabcut_roi_percent,
    banana_opening_kernel=int(settings["banana_opening"]),
    banana_closing_kernel=int(settings["banana_closing"]),
    blemish_opening_kernel=int(settings["blemish_opening"]),
    blemish_closing_kernel=int(settings["blemish_closing"]),
    config=config,
)

render_results(
    image,
    result,
    is_mode_a=is_mode_a,
    banana_opening=int(settings["banana_opening"]),
    banana_closing=int(settings["banana_closing"]),
    blemish_opening=int(settings["blemish_opening"]),
    blemish_closing=int(settings["blemish_closing"]),
)
