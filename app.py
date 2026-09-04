"""Streamlit entry point for banana surface-quality and ripeness assessment."""

from __future__ import annotations

from dataclasses import replace

import streamlit as st

# Streamlit requires page configuration before the rest of the UI is created.
st.set_page_config(page_title="Banana Quality & Ripeness Assessment", layout="wide")

from banana_quality.config import ProcessingConfig
from banana_quality.pipeline import analyse_image
from streamlit_ui.controls import render_sidebar
from streamlit_ui.input_section import render_image_input
from streamlit_ui.results import render_results
from streamlit_ui.ripeness import classify_ripeness_for_ui


st.title("Banana Quality and Ripeness Assessment")
st.caption(
    "K-means creates one shared banana mask using the current sidebar K value. "
    "That same mask is used for blemish analysis and to prepare the banana "
    "image for Teachable Machine ripeness classification."
)

settings = render_sidebar()

image, input_source = render_image_input()

if image is None:
    if input_source == "Upload image":
        st.info("Upload an image to start.")
    else:
        st.info(
            "The live preview can continue above. Press **Capture current frame** "
            "when the banana is positioned correctly. The captured frame will be "
            "used for both surface analysis and ripeness classification."
        )
    st.stop()

config = replace(
    ProcessingConfig(),
    banana_cluster_score_threshold=float(settings["banana_score_threshold"]),
    blemish_kmeans_k=int(settings["blemish_k"]),
    blemish_cluster_chroma_threshold=float(settings["blemish_chroma_threshold"]),
    darkness_threshold=float(settings["darkness_threshold"]),
)

with st.spinner("Analysing banana..."):
    result = analyse_image(
        image,
        kmeans_k=int(settings["kmeans_k"]),
        banana_opening_kernel=int(settings["banana_opening"]),
        banana_closing_kernel=int(settings["banana_closing"]),
        blemish_opening_kernel=int(settings["blemish_opening"]),
        blemish_closing_kernel=int(settings["blemish_closing"]),
        config=config,
    )

    # Reuse the exact final banana mask produced above.  The K slider therefore
    # affects one shared mask that feeds BOTH blemish and ripeness analysis.
    ripeness_result = classify_ripeness_for_ui(
        image,
        banana_mask=result["banana_mask"],
    )

render_results(
    image,
    result,
    ripeness_result=ripeness_result,
    banana_opening=int(settings["banana_opening"]),
    banana_closing=int(settings["banana_closing"]),
    blemish_opening=int(settings["blemish_opening"]),
    blemish_closing=int(settings["blemish_closing"]),
)
