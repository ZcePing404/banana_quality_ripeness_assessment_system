"""Second-stage blemish detection inside the final banana mask."""

from __future__ import annotations

import cv2
import numpy as np

from ..config import ProcessingConfig
from ..image_utils import odd_kernel
from ..morphology import _morphology, _remove_tiny_components

def blemish_mask_from_prepared(
    prepared: dict[str, object],
    banana_mask: np.ndarray,
    opening_kernel_size: int = 3,
    closing_kernel_size: int = 5,
    config: ProcessingConfig | None = None,
) -> dict[str, object]:
    """Detect blemish regions inside one segmented banana.

    Main detector
    -------------
    A second K-means is run ONLY on pixels inside the banana. The clustering
    features are CIELAB a,b, deliberately excluding L from K-means so gradual
    brightness changes caused by shadow influence the clustering much less.

    The healthiest-looking colour-cluster family becomes the adaptive healthy
    peel reference. Other clusters are classified by a,b chroma distance from
    that healthy family.

    Supplemental Rule A/B are retained for subtle damaged pixels that may stay
    inside a broad K-means cluster. The previous darkness-only Rule C is removed.
    """
    config = config or ProcessingConfig()
    blurred = np.asarray(prepared["blurred"], dtype=np.uint8)
    banana = np.where(banana_mask > 0, 255, 0).astype(np.uint8)

    opening_kernel = odd_kernel(opening_kernel_size)
    closing_kernel = odd_kernel(closing_kernel_size)

    if not np.any(banana):
        zero = np.zeros_like(banana)
        zero3 = np.zeros((*banana.shape, 3), dtype=np.uint8)
        return {
            "inner_banana": zero,
            "healthy_peel_candidates": zero,
            "blemish_kmeans_quantised_bgr": zero3,
            "blemish_cluster_distance_map": zero,
            "kmeans_region_candidate": zero,
            "bright_highlight_mask": zero,
            "rule_a_mask": zero,
            "rule_b_mask": zero,
            "raw_blemish_candidate": zero,
            "blemish_opened": zero,
            "blemish_closed": zero,
            "blemish_mask": zero,
            "blemish_cluster_summary": [],
            "healthy_cluster_ids": [],
            "blemish_cluster_ids": [],
            "blemish_opening_kernel": opening_kernel,
            "blemish_closing_kernel": closing_kernel,
        }

    # Ignore a very thin outer peel boundary because object edges naturally
    # become darker. If erosion leaves too little banana, use the whole mask.
    inner_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    inner = cv2.erode(banana, inner_kernel, iterations=1)
    if int((inner > 0).sum()) < 100:
        inner = banana.copy()

    filtered = cv2.medianBlur(blurred, 5)
    lab = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV).astype(np.float32)

    lightness, a_channel, b_channel = cv2.split(lab)
    hue, saturation, value = cv2.split(hsv)
    inside = inner > 0

    # ------------------------------------------------------------------
    # SECOND-STAGE K-MEANS: banana pixels only, using Lab a,b only.
    # ------------------------------------------------------------------
    inside_pixels = int(inside.sum())
    requested_k = max(2, int(config.blemish_kmeans_k))
    k = min(requested_k, inside_pixels)

    ab_pixels = np.column_stack(
        (a_channel[inside], b_channel[inside])
    ).astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        40,
        0.5,
    )
    cv2.setRNGSeed(42)
    _, labels_flat, centres_ab = cv2.kmeans(
        ab_pixels,
        int(k),
        None,
        criteria,
        5,
        cv2.KMEANS_PP_CENTERS,
    )
    labels_flat = labels_flat.ravel()

    cluster_label_map = np.full(inner.shape, -1, dtype=np.int16)
    cluster_label_map[inside] = labels_flat.astype(np.int16)

    # Build a cluster summary. "Healthy score" rewards stronger yellow b and
    # greener/lower a. L is deliberately NOT included so shadows are not
    # automatically treated as a different peel type.
    cluster_summary: list[dict[str, float | int | bool]] = []
    total_inside = max(1, inside_pixels)

    for cluster_id in range(int(k)):
        member = labels_flat == cluster_id
        count_pixels = int(member.sum())
        fraction = float(count_pixels / total_inside)

        if count_pixels > 0:
            mean_l = float(lightness[inside][member].mean())
            mean_a = float(a_channel[inside][member].mean())
            mean_b = float(b_channel[inside][member].mean())
            mean_sat = float(saturation[inside][member].mean())
            mean_value = float(value[inside][member].mean())
        else:
            mean_l = mean_a = mean_b = mean_sat = mean_value = 0.0

        healthy_score = float(mean_b - 0.80 * mean_a)

        cluster_summary.append(
            {
                "cluster": int(cluster_id),
                "pixels": count_pixels,
                "fraction": fraction,
                "mean_L": mean_l,
                "mean_a": mean_a,
                "mean_b": mean_b,
                "mean_saturation": mean_sat,
                "mean_value": mean_value,
                "healthy_score": healthy_score,
                "healthy_reference": False,
                "chroma_distance_to_healthy": 0.0,
                "blemish_cluster": False,
            }
        )

    # Avoid choosing a tiny specular/noise cluster as the healthy reference.
    eligible = [
        row
        for row in cluster_summary
        if float(row["fraction"]) >= float(config.healthy_cluster_min_fraction)
    ]
    if not eligible:
        eligible = cluster_summary

    best_healthy_score = max(float(row["healthy_score"]) for row in eligible)

    # Use a FAMILY of healthy-looking clusters rather than only one cluster.
    # This helps green and yellow healthy peel coexist on the same banana.
    healthy_cluster_ids = [
        int(row["cluster"])
        for row in eligible
        if float(row["healthy_score"])
        >= best_healthy_score - float(config.healthy_cluster_score_tolerance)
    ]

    if not healthy_cluster_ids:
        healthy_cluster_ids = [
            int(max(eligible, key=lambda row: float(row["healthy_score"]))["cluster"])
        ]

    healthy_centres = np.asarray(
        [centres_ab[cid] for cid in healthy_cluster_ids],
        dtype=np.float32,
    )

    # Compute each cluster's distance to its nearest healthy-family centre.
    blemish_cluster_ids: list[int] = []
    cluster_distances = np.zeros(int(k), dtype=np.float32)

    for row in cluster_summary:
        cluster_id = int(row["cluster"])
        centre = centres_ab[cluster_id].astype(np.float32)
        distances = np.linalg.norm(healthy_centres - centre, axis=1)
        nearest_distance = float(distances.min()) if len(distances) else 0.0
        cluster_distances[cluster_id] = nearest_distance

        is_healthy = cluster_id in healthy_cluster_ids
        is_blemish = (
            (not is_healthy)
            and nearest_distance >= float(config.blemish_cluster_chroma_threshold)
        )

        row["healthy_reference"] = bool(is_healthy)
        row["chroma_distance_to_healthy"] = nearest_distance
        row["blemish_cluster"] = bool(is_blemish)

        if is_blemish:
            blemish_cluster_ids.append(cluster_id)

    healthy = inside & np.isin(cluster_label_map, healthy_cluster_ids)

    # Median reference of all pixels in the selected healthy cluster family.
    healthy_colour = np.median(lab[healthy], axis=0)
    healthy_hue = float(np.median(hue[healthy]))

    # ------------------------------------------------------------------
    # REGION CANDIDATE FROM SECOND K-MEANS
    # ------------------------------------------------------------------
    region_candidate = inside & np.isin(
        cluster_label_map,
        blemish_cluster_ids,
    )

    # Bright glare can have very low/neutral a,b and otherwise look like a
    # strong chroma change. Reject obvious bright low/moderate-saturation glare.
    bright_highlight = inside & (
        (value >= float(config.highlight_value_threshold))
        & (saturation <= float(config.highlight_saturation_threshold))
    )
    region_candidate &= ~bright_highlight

    # ------------------------------------------------------------------
    # SUPPLEMENTAL RULE A/B
    # ------------------------------------------------------------------
    darkness = np.maximum(healthy_colour[0] - lightness, 0.0)
    redness = np.maximum(a_channel - healthy_colour[1], 0.0)
    yellow_loss = np.maximum(healthy_colour[2] - b_channel, 0.0)

    # Circular OpenCV hue difference (0..179). This detects change in either
    # direction and handles wraparound at red.
    hue_difference = np.abs(hue - healthy_hue)
    hue_difference = np.minimum(hue_difference, 180.0 - hue_difference)

    valid_colour = saturation >= float(config.blemish_min_saturation)

    rule_a = inside & ~bright_highlight & (
        valid_colour
        & (hue_difference > float(config.hue_drop_threshold))
        & (darkness > float(config.darkness_threshold))
    )

    rule_b = inside & ~bright_highlight & (
        valid_colour
        & (hue_difference > float(config.strong_hue_drop_threshold))
        & (
            (redness > float(config.redness_threshold))
            | (yellow_loss > float(config.yellow_loss_threshold))
        )
    )

    # IMPORTANT: the old pixel-by-pixel darkness-only Rule C is intentionally
    # removed. Dark/black regions are now recovered by the a,b region K-means
    # when their chroma differs from the healthy cluster family.
    candidate = region_candidate | rule_a | rule_b

    raw = np.where(candidate, 255, 0).astype(np.uint8)

    # Morphology is cleanup only.
    opened = _morphology(raw, cv2.MORPH_OPEN, opening_kernel)
    closed = _morphology(opened, cv2.MORPH_CLOSE, closing_kernel)

    minimum_area = max(
        8,
        int((banana > 0).sum() * float(config.min_blemish_area_ratio)),
    )
    final_mask = _remove_tiny_components(closed, minimum_area)
    final_mask[banana == 0] = 0

    # ------------------------------------------------------------------
    # DEBUG/VISUALISATION OUTPUTS
    # ------------------------------------------------------------------
    # Recolour only a,b by the second-stage centres while preserving L.
    # This lets the user see chroma groups without pretending L was clustered.
    quant_lab = lab.copy()
    for cluster_id in range(int(k)):
        cluster_mask = cluster_label_map == cluster_id
        quant_lab[:, :, 1][cluster_mask] = centres_ab[cluster_id, 0]
        quant_lab[:, :, 2][cluster_mask] = centres_ab[cluster_id, 1]

    quant_lab_u8 = np.clip(quant_lab, 0, 255).astype(np.uint8)
    quantised_bgr = cv2.cvtColor(quant_lab_u8, cv2.COLOR_LAB2BGR)
    quantised_bgr[~inside] = 0

    distance_map = np.zeros_like(lightness, dtype=np.float32)
    for cluster_id in range(int(k)):
        distance_map[cluster_label_map == cluster_id] = cluster_distances[cluster_id]

    max_distance = float(distance_map[inside].max()) if np.any(inside) else 0.0
    if max_distance > 0:
        distance_display = np.clip(distance_map / max_distance * 255.0, 0, 255).astype(np.uint8)
    else:
        distance_display = np.zeros_like(banana)

    return {
        "inner_banana": inner,
        "healthy_peel_candidates": np.where(healthy, 255, 0).astype(np.uint8),

        # Second-stage region/K-means diagnostics.
        "blemish_kmeans_quantised_bgr": quantised_bgr,
        "blemish_cluster_distance_map": distance_display,
        "kmeans_region_candidate": np.where(region_candidate, 255, 0).astype(np.uint8),
        "bright_highlight_mask": np.where(bright_highlight, 255, 0).astype(np.uint8),
        "blemish_cluster_summary": cluster_summary,
        "healthy_cluster_ids": healthy_cluster_ids,
        "blemish_cluster_ids": blemish_cluster_ids,

        # Supplemental rule diagnostics.
        "rule_a_mask": np.where(rule_a, 255, 0).astype(np.uint8),
        "rule_b_mask": np.where(rule_b, 255, 0).astype(np.uint8),

        "raw_blemish_candidate": raw,
        "blemish_opened": opened,
        "blemish_closed": closed,
        "blemish_mask": final_mask,

        # Numeric debug maps retained for inspection.
        "lightness_map": lightness,
        "saturation_map": saturation,
        "value_map": value,
        "darkness_map": darkness,
        "redness_map": redness,
        "yellow_loss_map": yellow_loss,
        "hue_difference_map": hue_difference,
        # Backward-compatible alias used by older app/debug code.
        "hue_drop_map": hue_difference,

        "blemish_opening_kernel": opening_kernel,
        "blemish_closing_kernel": closing_kernel,
    }

