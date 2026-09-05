"""Sidebar defaults and reusable Streamlit controls."""

from __future__ import annotations

import streamlit as st

from banana_quality.config import ProcessingConfig


def load_defaults() -> dict[str, int | float]:
    """Read Streamlit starting values from ProcessingConfig."""
    config = ProcessingConfig()

    return {
        "kmeans_k": int(config.kmeans_k),
        "banana_cluster_score_threshold": float(
            config.banana_cluster_score_threshold
        ),
        "banana_opening_kernel": int(config.banana_opening_kernel),
        "banana_closing_kernel": int(config.banana_closing_kernel),
        "blemish_kmeans_k": int(config.blemish_kmeans_k),
        "blemish_cluster_chroma_threshold": float(
            config.blemish_cluster_chroma_threshold
        ),
        "blemish_opening_kernel": int(config.blemish_opening_kernel),
        "blemish_closing_kernel": int(config.blemish_closing_kernel),
    }


def kernel_slider(
    label: str,
    default: int,
    help_text: str,
) -> int:
    """Render an odd-sized morphology kernel slider."""

    return st.slider(
        label,
        1,
        21,
        default,
        2,
        help=help_text,
    )


def render_sidebar() -> dict[str, object]:
    """Render controls for the two-stage K-means pipeline."""

    definitions = load_defaults()

    with st.sidebar:

        # ========================================================
        # BANANA SEGMENTATION
        # ========================================================

        st.header("Banana Segmentation")
        st.caption("Automatic K-means segmentation")

        st.divider()

        st.subheader("Banana Mask Settings")

        kmeans_k = st.slider(
            "K-means colour clusters (K)",
            3,
            30,
            int(definitions["kmeans_k"]),
            1,
            help=(
                "Sets the number of Lab colour groups used to separate "
                "the banana from the image.\n\n"
                "Increase: finer colour separation, but the mask may "
                "become fragmented.\n\n"
                "Decrease: broader colour groups, but banana and similar "
                "background colours may merge."
            ),
        )

        banana_score_threshold = st.slider(
            "Banana cluster acceptance",
            0.20,
            0.80,
            float(definitions["banana_cluster_score_threshold"]),
            0.05,
            help=(
                "Sets how banana-like a K-means cluster must be before "
                "it is included in the banana mask.\n\n"
                "Increase: stricter selection and less background, but "
                "some true banana regions may be removed.\n\n"
                "Decrease: more permissive selection, but background "
                "may be included."
            ),
        )

        banana_opening = kernel_slider(
            "Banana opening kernel",
            int(definitions["banana_opening_kernel"]),
            (
                "Removes small foreground noise from the banana mask. "
                "1x1 disables opening.\n\n"
                "Increase: removes larger noisy regions but may erase "
                "thin banana parts."
            ),
        )

        banana_closing = kernel_slider(
            "Banana closing kernel",
            int(definitions["banana_closing_kernel"]),
            (
                "Fills small gaps and reconnects broken regions in the "
                "banana mask. 1x1 disables closing.\n\n"
                "Increase: fills larger gaps but may connect unwanted regions."
            ),
        )

        # ========================================================
        # BLEMISH DETECTION
        # ========================================================

        st.divider()

        st.header("Blemish Detection")
        st.caption(
            "Second-stage Lab a,b K-means + highlight exclusion + morphology"
        )

        blemish_k = st.slider(
            "Blemish colour clusters (K)",
            3,
            10,
            int(definitions["blemish_kmeans_k"]),
            1,
            help=(
                "Sets the number of Lab a,b colour groups used inside "
                "the banana.\n\n"
                "Increase: separates finer colour differences but may "
                "create fragmented/noisy regions.\n\n"
                "Decrease: creates broader groups but subtle blemishes "
                "may merge with healthy peel."
            ),
        )

        blemish_chroma_threshold = st.slider(
            "Blemish colour sensitivity",
            8.0,
            35.0,
            float(definitions["blemish_cluster_chroma_threshold"]),
            1.0,
            help=(
                "Sets how different a colour cluster must be from the "
                "healthy peel reference to become a blemish candidate.\n\n"
                "Increase: stricter detection and fewer false positives.\n\n"
                "Decrease: more sensitive detection, but more false "
                "positives may occur."
            ),
        )

        blemish_opening = kernel_slider(
            "Blemish opening kernel",
            int(definitions["blemish_opening_kernel"]),
            (
                "Removes small isolated regions from the K-means blemish mask. "
                "1x1 disables opening.\n\n"
                "Increase: removes more small noise but may remove genuine "
                "small blemishes."
            ),
        )

        blemish_closing = kernel_slider(
            "Blemish closing kernel",
            int(definitions["blemish_closing_kernel"]),
            (
                "Reconnects fragmented blemish regions and fills small gaps. "
                "1x1 disables closing.\n\n"
                "Increase: connects larger gaps but may merge nearby regions."
            ),
        )

    return {
        "kmeans_k": kmeans_k,
        "banana_score_threshold": banana_score_threshold,
        "banana_opening": banana_opening,
        "banana_closing": banana_closing,
        "blemish_k": blemish_k,
        "blemish_chroma_threshold": blemish_chroma_threshold,
        "blemish_opening": blemish_opening,
        "blemish_closing": blemish_closing,
    }