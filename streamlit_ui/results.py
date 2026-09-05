"""Result and simplified processing-detail panels for Streamlit."""

from __future__ import annotations

import cv2
import streamlit as st

from banana_quality.visualisation import overlay_result


def render_results(
    image,
    result,
    *,
    ripeness_result: dict[str, object] | None = None,
    banana_opening: int,
    banana_closing: int,
    blemish_opening: int,
    blemish_closing: int,
) -> None:
    """Render the main assessment outputs and only the key processing stages.

    The detailed development/debug views (Rule A, Rule B, distance maps,
    morphology stages, and cluster tables) are intentionally not shown in
    the normal application UI. They can still remain inside the processing
    pipeline for report figures or debugging when needed.
    """
    prepared = result["prepared"]
    blemish_steps = result["blemish_steps"]
    overlay = overlay_result(
        image,
        result["banana_mask"],
        result["blemish_mask"],
    )

    # ------------------------------------------------------------
    # Main numerical results
    # ------------------------------------------------------------
    st.subheader("Analysis results")

    m1, m2, m3, m4 = st.columns(4)

    if ripeness_result and bool(ripeness_result.get("available")):
        ripeness_name = str(ripeness_result.get("class_name", "Unknown"))
        confidence = float(ripeness_result.get("confidence", 0.0))
        m1.metric("Ripeness", ripeness_name)
        m2.metric("Ripeness confidence", f"{confidence * 100:.1f}%")
    else:
        m1.metric("Ripeness", "Unavailable")
        m2.metric("Ripeness confidence", "—")

    m3.metric(
        "Visible blemish percentage",
        f"{result['blemish_percentage']:.2f}%",
    )
    m4.metric("Surface quality", result["quality_grade"])

    if not ripeness_result or not bool(ripeness_result.get("available")):
        message = (
            str(ripeness_result.get("error"))
            if ripeness_result
            else "Ripeness classification result is unavailable."
        )
        st.warning(message)

    # ------------------------------------------------------------
    # Final outputs
    # ------------------------------------------------------------
    st.markdown("### Final output")

    c1, c2, c3, c4 = st.columns(4)

    c1.image(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        caption="Input image",
        use_container_width=True,
    )
    c2.image(
        result["banana_mask"],
        caption="Final banana mask",
        use_container_width=True,
    )
    c3.image(
        result["blemish_mask"],
        caption="Final blemish mask",
        use_container_width=True,
    )
    c4.image(
        cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
        caption="Detection overlay",
        use_container_width=True,
    )

    st.caption(
        "The final banana mask defines the visible banana region used for "
        "surface-quality analysis. The blemish percentage is calculated from "
        "the final blemish mask relative to this visible banana region."
    )

    # ------------------------------------------------------------
    # Optional ripeness details — collapsed by default
    # ------------------------------------------------------------
    if ripeness_result and bool(ripeness_result.get("available")):
        with st.expander("Ripeness classification details", expanded=False):
            segmented_image = ripeness_result.get("segmented_image")

            if segmented_image is not None:
                st.image(
                    cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB),
                    caption="Image sent to the ripeness classifier",
                    use_container_width=True,
                )

            probabilities = dict(ripeness_result.get("probabilities", {}))

            if probabilities:
                st.markdown("#### Class probabilities")

                for class_name, probability in sorted(
                    probabilities.items(),
                    key=lambda item: float(item[1]),
                    reverse=True,
                ):
                    probability = max(0.0, min(1.0, float(probability)))
                    label_col, bar_col, value_col = st.columns([1.3, 4.0, 0.9])

                    label_col.write(str(class_name))
                    bar_col.progress(probability)
                    value_col.write(f"{probability * 100:.1f}%")

    # ------------------------------------------------------------
    # Simplified processing details
    # ------------------------------------------------------------
    with st.expander("Processing details", expanded=False):
        st.caption(
            "Only the main stages are shown here. Detailed Rule A / Rule B, "
            "cluster-distance, highlight-exclusion, and morphology debug views "
            "are intentionally hidden from the normal user interface."
        )

        # ============================
        # Banana segmentation
        # ============================
        st.markdown("#### Banana segmentation")

        b1, b2, b3 = st.columns(3)

        b1.image(
            cv2.cvtColor(prepared["working_image"], cv2.COLOR_BGR2RGB),
            caption="1. Input / standardised image",
            use_container_width=True,
        )
        b2.image(
            prepared["raw_banana_candidate"],
            caption="2. K-means banana candidate",
            use_container_width=True,
        )
        b3.image(
            result["banana_mask"],
            caption="3. Final banana mask",
            use_container_width=True,
        )

        st.caption(
            "K-means first identifies banana-like colour regions. "
            "Morphological refinement and connected-component processing then "
            "produce the final shared banana mask."
        )

        # ============================
        # Blemish detection
        # ============================
        st.markdown("#### Blemish detection")

        r1, r2, r3 = st.columns(3)

        r1.image(
            blemish_steps["inner_banana"],
            caption="1. Analysis area (banana mask)",
            use_container_width=True,
        )
        r2.image(
            blemish_steps["kmeans_region_candidate"],
            caption="2. K-means blemish candidate",
            use_container_width=True,
        )
        r3.image(
            blemish_steps["raw_blemish_candidate"],
            caption="3. Combined blemish candidate",
            use_container_width=True,
        )

        r4, r5 = st.columns(2)

        r4.image(
            result["blemish_mask"],
            caption="4. Final blemish mask",
            use_container_width=True,
        )
        r5.image(
            cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
            caption="5. Detection overlay",
            use_container_width=True,
        )

        st.caption(
            "The K-means candidate is supplemented by the internal colour-based "
            "rules, while detected bright highlights are excluded. These results "
            "form the combined blemish candidate. Morphological refinement and "
            "small-region filtering then produce the final blemish mask."
        )

        st.info(
            "If a false positive or false negative needs to be investigated, "
            "the detailed internal masks (healthy-cluster family, colour-distance "
            "map, highlight exclusion, Rule A, Rule B, opening and closing) can "
            "still be inspected during development or shown selectively in the report."
        )
