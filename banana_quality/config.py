from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProcessingConfig:
    # ---------- Common preprocessing ----------
    max_dimension: int = 416
    gaussian_kernel: int = 5

    # ---------- First K-means: whole-image banana segmentation ----------
    kmeans_k: int = 7
    banana_cluster_score_threshold: float = 0.35

    # Broad banana-colour prior used only to decide which K-means clusters
    # are likely to belong to the banana. OpenCV hue uses 0..179.
    banana_hue_min: int = 4
    banana_hue_max: int = 85
    banana_min_saturation: int = 45
    banana_min_value: int = 45
    banana_min_lab_b: int = 128

    # ---------- Banana-mask morphology ----------
    # Starting values for the Streamlit controls and formal evaluation.
    banana_opening_kernel: int = 3
    banana_closing_kernel: int = 7

    # ---------- Second K-means: blemish detection ----------
    # 1) Run K-means only inside the final banana mask.
    # 2) Cluster using CIELAB a,b so brightness/shadow has less influence.
    # 3) Select a family of healthy-looking peel clusters as the reference.
    # 4) Mark clusters with sufficient a,b chroma distance as blemish candidates.
    # 5) Exclude obvious bright highlights.
    # 6) Refine the candidate with morphology and small-component removal.
    blemish_kmeans_k: int = 6
    healthy_cluster_min_fraction: float = 0.05
    healthy_cluster_score_tolerance: float = 18.0
    blemish_cluster_chroma_threshold: float = 21.0

    # Bright-highlight exclusion.
    highlight_value_threshold: float = 220.0
    highlight_saturation_threshold: float = 100.0

    # Final small-component cleanup.
    min_blemish_area_ratio: float = 0.0002

    # Blemish-mask morphology.
    blemish_opening_kernel: int = 3
    blemish_closing_kernel: int = 5
