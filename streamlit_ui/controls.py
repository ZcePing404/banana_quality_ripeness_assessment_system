"""Sidebar defaults and reusable Streamlit controls."""

from __future__ import annotations

import streamlit as st

from banana_quality.config import ProcessingConfig


def load_defaults() -> dict[str, int | float]:
    """Read Streamlit starting values from ``ProcessingConfig``."""
    config = ProcessingConfig()
    return {
        "kmeans_k": int(config.kmeans_k),
        "banana_cluster_score_threshold": float(config.banana_cluster_score_threshold),
        "banana_opening_kernel": int(config.banana_opening_kernel),
        "banana_closing_kernel": int(config.banana_closing_kernel),
        "blemish_kmeans_k": int(config.blemish_kmeans_k),
        "blemish_cluster_chroma_threshold": float(config.blemish_cluster_chroma_threshold),
        "darkness_threshold": float(config.darkness_threshold),
        "blemish_opening_kernel": int(config.blemish_opening_kernel),
        "blemish_closing_kernel": int(config.blemish_closing_kernel),
    }


def kernel_slider(label: str, default: int, help_text: str) -> int:
    """Render an odd-sized morphology kernel slider."""
    return st.slider(label, 1, 21, default, 2, help=help_text)


def render_sidebar() -> dict[str, object]:
    """Render K-means banana-mask and blemish controls."""
    definitions = load_defaults()

    with st.sidebar:
        st.header("Banana Segmentation")
        st.caption("Automatic K-means segmentation")

        st.divider()
        st.header("Banana Mask Settings")

        kmeans_k = st.slider(
            "K-means colour clusters (K)",
            3,
            30,
            int(definitions["kmeans_k"]),
            1,
            help=(
                "Usage: Sets the number of Lab colour groups used to separate the banana from the image.\n\n"
                "Increase: Finer colour separation, but the mask may become fragmented.\n\n"
                "Decrease: Broader colour groups, but banana and similar background colours may merge."
            ),
        )
        banana_score_threshold = st.slider(
            "Banana cluster acceptance",
            0.20,
            0.80,
            float(definitions["banana_cluster_score_threshold"]),
            0.05,
            help=(
                "Usage: Sets how banana-like a K-means cluster must be before it is included in the banana mask.\n\n"
                "Increase: Stricter selection; removes more background but may remove real banana regions.\n\n"
                "Decrease: More permissive selection; keeps more banana regions but may include background."
            ),
        )
        banana_opening = kernel_slider(
            "Banana opening kernel",
            int(definitions["banana_opening_kernel"]),
            (
                "Usage: Removes small foreground noise from the banana mask. 1x1 disables opening.\n\n"
                "Increase: Removes larger small/noisy regions but may erase thin banana parts.\n\n"
                "Decrease: Preserves more banana detail but also keeps more noise."
            ),
        )
        banana_closing = kernel_slider(
            "Banana closing kernel",
            int(definitions["banana_closing_kernel"]),
            (
                "Usage: Fills small gaps and reconnects broken areas in the banana mask. 1x1 disables closing.\n\n"
                "Increase: Fills larger gaps but may join unwanted nearby regions.\n\n"
                "Decrease: Changes the mask less but may leave small gaps or holes."
            ),
        )

        st.divider()
        st.header("Blemish Settings")

        blemish_k = st.slider(
            "Blemish colour clusters (K)",
            3,
            10,
            int(definitions["blemish_kmeans_k"]),
            1,
            help=(
                "Usage: Sets the number of Lab a,b colour groups used inside the banana for blemish analysis.\n\n"
                "Increase: Separates finer colour differences but may create fragmented/noisy regions.\n\n"
                "Decrease: Creates broader groups but subtle blemishes may merge with healthy peel."
            ),
        )
        blemish_chroma_threshold = st.slider(
            "Blemish colour sensitivity",
            8.0,
            35.0,
            float(definitions["blemish_cluster_chroma_threshold"]),
            1.0,
            help=(
                "Usage: Sets how different a colour cluster must be from healthy peel to become a blemish candidate.\n\n"
                "Increase: Stricter detection; fewer false positives but subtle blemishes may be missed.\n\n"
                "Decrease: More sensitive detection; finds smaller colour changes but may add false positives."
            ),
        )
        # Keep the tuned darkness threshold fixed internally instead of exposing
        # it as a user-facing control. app.py can continue reading the same key.
        darkness_threshold = float(definitions["darkness_threshold"])
        blemish_opening = kernel_slider(
            "Blemish opening kernel",
            int(definitions["blemish_opening_kernel"]),
            (
                "Usage: Removes small isolated noise from the blemish mask. 1x1 disables opening.\n\n"
                "Increase: Removes more small regions but may erase genuine tiny blemishes.\n\n"
                "Decrease: Preserves smaller blemishes but also keeps more noise."
            ),
        )
        blemish_closing = kernel_slider(
            "Blemish closing kernel",
            int(definitions["blemish_closing_kernel"]),
            (
                "Usage: Fills gaps and joins fragmented areas in the blemish mask. 1x1 disables closing.\n\n"
                "Increase: Connects larger gaps but may merge separate blemishes.\n\n"
                "Decrease: Preserves separated regions but may leave fragmented blemish areas."
            ),
        )

    return {
        "kmeans_k": kmeans_k,
        "banana_score_threshold": banana_score_threshold,
        "banana_opening": banana_opening,
        "banana_closing": banana_closing,
        "blemish_k": blemish_k,
        "blemish_chroma_threshold": blemish_chroma_threshold,
        "darkness_threshold": darkness_threshold,
        "blemish_opening": blemish_opening,
        "blemish_closing": blemish_closing,
    }
