"""Configuration values for the classical banana image-processing pipeline."""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class ProcessingConfig:
    # ---------- Common preprocessing ----------
    max_dimension: int = 416
    gaussian_kernel: int = 5

    # ---------- Automatic whole-image K-means ----------
    kmeans_k: int = 7
    banana_cluster_score_threshold: float = 0.35

    # Broad banana-colour prior used only to decide which K-means clusters
    # are likely to belong to banana. OpenCV hue uses 0..179.
    banana_hue_min: int = 4
    banana_hue_max: int = 85
    banana_min_saturation: int = 45
    banana_min_value: int = 45
    banana_min_lab_b: int = 128

    # ---------- Mode A morphology defaults ----------
    # These values are used as the starting positions of the Streamlit sliders.
    # They remain user-adjustable during interactive analysis.
    banana_opening_kernel: int = 3
    banana_closing_kernel: int = 7

    # ---------- Blemish detection ----------
    # Main idea:
    #   1) Run a SECOND K-means only inside the final banana mask.
    #   2) Cluster mainly with CIELAB a,b so brightness/shadow (L) has much
    #      less influence on the grouping.
    #   3) Use the healthiest-looking cluster family as the colour reference.
    #   4) Clusters whose a,b chroma is sufficiently different become region
    #      blemish candidates.
    #   5) Rule A/B remain as conservative supplemental pixel evidence.
    #
    # These are fixed defaults for formal evaluation. Do not tune them
    # separately for each test image when reporting IoU/Dice.
    blemish_kmeans_k: int = 6
    healthy_cluster_min_fraction: float = 0.05
    healthy_cluster_score_tolerance: float = 18.0
    blemish_cluster_chroma_threshold: float = 21.0

    # Supplemental pixel evidence.
    darkness_threshold: float = 36.0
    hue_drop_threshold: float = 7.0
    strong_hue_drop_threshold: float = 10.0
    redness_threshold: float = 11.0
    yellow_loss_threshold: float = 20.0

    # Hue is unstable in low-saturation p   ixels, while bright highlights can
    # lose yellow chroma and otherwise look like damage.
    blemish_min_saturation: float = 35.0
    highlight_value_threshold: float = 220.0
    highlight_saturation_threshold: float = 100.0

    # Final small-component cleanup.
    min_blemish_area_ratio: float = 0.0002

    # Morphology defaults used as the starting positions of the Streamlit sliders.
    blemish_opening_kernel: int = 3
    blemish_closing_kernel: int = 3

