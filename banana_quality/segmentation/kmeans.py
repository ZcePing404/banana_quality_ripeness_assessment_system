"""Automatic whole-image K-means banana segmentation preparation (Mode A)."""

from __future__ import annotations

import cv2
import numpy as np

from ..config import ProcessingConfig
from ..preprocessing import _common_preprocess

def prepare_image(
    image_bgr: np.ndarray,
    config: ProcessingConfig | None = None,
) -> dict[str, object]:
    """Prepare AUTOMATIC whole-image K-means banana segmentation."""
    config = config or ProcessingConfig()
    prepared = _common_preprocess(image_bgr, config)

    blurred = np.asarray(prepared["blurred"], dtype=np.uint8)
    lab_u8 = np.asarray(prepared["lab"], dtype=np.uint8)
    hsv = np.asarray(prepared["hsv"], dtype=np.uint8)

    height, width = lab_u8.shape[:2]
    pixels = lab_u8.reshape(-1, 3).astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        1.0,
    )
    cv2.setRNGSeed(42)
    _, labels_flat, centres = cv2.kmeans(
        pixels,
        int(config.kmeans_k),
        None,
        criteria,
        3,
        cv2.KMEANS_PP_CENTERS,
    )
    labels = labels_flat.reshape(height, width)

    # Rebuild a quantised image so the user can see the K-means colour groups.
    quantised_lab = centres[labels_flat.ravel()].reshape(height, width, 3)
    quantised_lab = np.clip(quantised_lab, 0, 255).astype(np.uint8)
    quantised_bgr = cv2.cvtColor(quantised_lab, cv2.COLOR_LAB2BGR)

    hue, saturation, value = cv2.split(hsv)
    _, _, lab_b = cv2.split(lab_u8)

    # Broad colour prior. This is NOT the final mask; it only scores K-means clusters.
    banana_colour_prior = (
        (hue >= config.banana_hue_min)
        & (hue <= config.banana_hue_max)
        & (saturation >= config.banana_min_saturation)
        & (value >= config.banana_min_value)
        & (lab_b >= config.banana_min_lab_b)
    )

    cluster_summary: list[dict[str, float | int | bool]] = []
    cluster_scores = np.zeros(int(config.kmeans_k), dtype=np.float32)

    for cluster_id in range(int(config.kmeans_k)):
        cluster_mask = labels == cluster_id
        pixel_count = int(cluster_mask.sum())

        if pixel_count == 0:
            banana_fraction = 0.0
            mean_hue = mean_sat = mean_value = mean_lab_b = 0.0
        else:
            # Example: 0.70 means 70% of this cluster's pixels satisfy
            # the broad banana-colour condition.
            banana_fraction = float(banana_colour_prior[cluster_mask].mean())
            mean_hue = float(hue[cluster_mask].mean())
            mean_sat = float(saturation[cluster_mask].mean())
            mean_value = float(value[cluster_mask].mean())
            mean_lab_b = float(lab_b[cluster_mask].mean())

        cluster_scores[cluster_id] = banana_fraction
        selected = banana_fraction >= config.banana_cluster_score_threshold
        cluster_summary.append(
            {
                "cluster": cluster_id,
                "pixels": pixel_count,
                "banana_like_fraction": banana_fraction,
                "mean_hue": mean_hue,
                "mean_saturation": mean_sat,
                "mean_value": mean_value,
                "mean_lab_b": mean_lab_b,
                "selected": bool(selected),
            }
        )

    selected_clusters = [
        int(row["cluster"])
        for row in cluster_summary
        if bool(row["selected"])
    ]

    # If nothing passes the threshold, keep the highest-scoring cluster so the
    # automatic mode does not return an empty candidate.
    if not selected_clusters:
        selected_clusters = [int(np.argmax(cluster_scores))]
        cluster_summary[selected_clusters[0]]["selected"] = True

    raw_banana = np.where(np.isin(labels, selected_clusters), 255, 0).astype(np.uint8)

    likelihood = cluster_scores[labels]
    likelihood_map = np.clip(likelihood * 255.0, 0, 255).astype(np.uint8)

    prepared.update(
        {
            "segmentation_mode": "kmeans",
            "kmeans_labels": labels,
            "kmeans_centres_lab": centres,
            "kmeans_quantised_bgr": quantised_bgr,
            "banana_colour_prior": np.where(banana_colour_prior, 255, 0).astype(np.uint8),
            "banana_likelihood_map": likelihood_map,
            "cluster_summary": cluster_summary,
            "selected_clusters": selected_clusters,
            "raw_banana_candidate": raw_banana,
        }
    )
    return prepared

