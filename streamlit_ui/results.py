"""Result, diagnostic-stage, and detail panels for Streamlit."""

from __future__ import annotations

import cv2
import pandas as pd
import streamlit as st

from banana_quality.visualisation import overlay_result


def render_results(image, result, *, ripeness_result: dict[str, object] | None = None,
                   is_mode_a: bool, banana_opening: int, banana_closing: int,
                   blemish_opening: int, blemish_closing: int) -> None:
    """Render ripeness, surface-quality outputs, processing stages and details."""
    prepared = result["prepared"]
    banana_steps = result["banana_steps"]
    blemish_steps = result["blemish_steps"]
    overlay = overlay_result(image, result["banana_mask"], result["blemish_mask"])

    # Main user-facing assessment combines the ML ripeness output with the
    # existing classical blemish/surface-quality output.
    m1, m2, m3, m4 = st.columns(4)

    if ripeness_result and bool(ripeness_result.get("available")):
        ripeness_name = str(ripeness_result.get("class_name", "Unknown"))
        confidence = float(ripeness_result.get("confidence", 0.0))
        m1.metric("Ripeness", ripeness_name)
        m2.metric("Ripeness confidence", f"{confidence * 100:.1f}%")
    else:
        m1.metric("Ripeness", "Unavailable")
        m2.metric("Ripeness confidence", "—")

    m3.metric("Blemish area", f"{result['blemish_percentage']:.2f}%")
    m4.metric("Surface quality", result["quality_grade"])

    if is_mode_a:
        st.caption(
            f"Banana segmentation: automatic K-means; selected clusters: "
            f"{prepared['selected_clusters']}"
        )
    else:
        st.caption("Banana segmentation: user ROI + GrabCut")

    if not ripeness_result or not bool(ripeness_result.get("available")):
        message = (
            str(ripeness_result.get("error"))
            if ripeness_result
            else "Ripeness classification result is unavailable."
        )
        st.warning(message)

    st.subheader("Final output")
    c1, c2, c3, c4 = st.columns(4)
    c1.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
    c2.image(result["banana_mask"], caption="Banana mask", use_container_width=True)
    c3.image(result["blemish_mask"], caption="Blemish mask", use_container_width=True)
    c4.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Overlay", use_container_width=True)

    if ripeness_result and bool(ripeness_result.get("available")):
        with st.expander("Ripeness prediction details", expanded=True):
            st.caption(
                "The Teachable Machine classifier uses the same full image supplied to "
                "the app. For the USB camera, this is the frozen frame created by "
                "**Capture current frame**."
            )

            probabilities = dict(ripeness_result.get("probabilities", {}))
            for class_name, probability in sorted(
                probabilities.items(),
                key=lambda item: float(item[1]),
                reverse=True,
            ):
                probability = max(0.0, min(1.0, float(probability)))
                label_col, bar_col, value_col = st.columns([1.2, 4.0, 0.9])
                label_col.write(str(class_name))
                bar_col.progress(probability)
                value_col.write(f"{probability * 100:.1f}%")

    with st.expander("Show processing stages", expanded=True):
        if is_mode_a:
            r1 = st.columns(4)
            r1[0].image(cv2.cvtColor(prepared["working_image"], cv2.COLOR_BGR2RGB), caption="1. Standardised image", use_container_width=True)
            r1[1].image(cv2.cvtColor(prepared["blurred"], cv2.COLOR_BGR2RGB), caption="2. Gaussian blur", use_container_width=True)
            r1[2].image(cv2.cvtColor(prepared["kmeans_quantised_bgr"], cv2.COLOR_BGR2RGB), caption=f"3. Whole-image K-means (K={result['kmeans_k']})", use_container_width=True)
            r1[3].image(prepared["banana_colour_prior"], caption="4. Banana-colour prior", use_container_width=True)
            r2 = st.columns(4)
            r2[0].image(prepared["banana_likelihood_map"], caption="5. Cluster banana-likelihood", use_container_width=True)
            r2[1].image(prepared["raw_banana_candidate"], caption="6. Raw K-means banana candidate", use_container_width=True)
            r2[2].image(banana_steps["banana_opened"], caption=f"7. Opening ({banana_opening}x{banana_opening})", use_container_width=True)
            r2[3].image(banana_steps["banana_closed"], caption=f"8. Closing ({banana_closing}x{banana_closing})", use_container_width=True)
        else:
            r1 = st.columns(4)
            r1[0].image(cv2.cvtColor(prepared["working_image"], cv2.COLOR_BGR2RGB), caption="1. Standardised image", use_container_width=True)
            r1[1].image(cv2.cvtColor(prepared["blurred"], cv2.COLOR_BGR2RGB), caption="2. Gaussian blur", use_container_width=True)
            r1[2].image(cv2.cvtColor(prepared["grabcut_rectangle_preview"], cv2.COLOR_BGR2RGB), caption="3. User-drawn ROI", use_container_width=True)
            r1[3].image(cv2.cvtColor(prepared["grabcut_initial_label_colour"], cv2.COLOR_BGR2RGB), caption="4. Initial GrabCut guidance", use_container_width=True)
            r2 = st.columns(4)
            r2[0].image(cv2.cvtColor(prepared["grabcut_final_label_colour"], cv2.COLOR_BGR2RGB), caption="5. GrabCut labels after optimisation", use_container_width=True)
            r2[1].image(prepared["raw_banana_candidate"], caption="6. Raw GrabCut foreground mask", use_container_width=True)
            r2[2].image(cv2.cvtColor(prepared["grabcut_foreground_preview"], cv2.COLOR_BGR2RGB), caption="7. Foreground isolated by GrabCut", use_container_width=True)
            r2[3].image(banana_steps["banana_mask"], caption="8. Final banana mask", use_container_width=True)
            st.caption("Mode B does not run K-means for banana segmentation and does not apply banana opening/closing morphology.")

        st.markdown("### Blemish processing: second-stage K-means inside banana")

        rb1 = st.columns(4)
        rb1[0].image(blemish_steps["inner_banana"], caption="1. Area allowed for blemish analysis", use_container_width=True)
        rb1[1].image(blemish_steps["healthy_peel_candidates"], caption="2. Healthy colour-cluster family", use_container_width=True)
        rb1[2].image(cv2.cvtColor(blemish_steps["blemish_kmeans_quantised_bgr"], cv2.COLOR_BGR2RGB), caption="3. Second K-means using Lab a,b", use_container_width=True)
        rb1[3].image(blemish_steps["blemish_cluster_distance_map"], caption="4. a,b distance from healthy clusters", use_container_width=True)

        rb2 = st.columns(4)
        rb2[0].image(blemish_steps["kmeans_region_candidate"], caption="5. Region blemish candidate", use_container_width=True)
        rb2[1].image(blemish_steps["bright_highlight_mask"], caption="6. Bright-highlight exclusion", use_container_width=True)
        rb2[2].image(blemish_steps["rule_a_mask"], caption="7. Supplemental Rule A", use_container_width=True)
        rb2[3].image(blemish_steps["rule_b_mask"], caption="8. Supplemental Rule B", use_container_width=True)

        rb3 = st.columns(4)
        rb3[0].image(blemish_steps["raw_blemish_candidate"], caption="9. Combined raw blemish candidate", use_container_width=True)
        rb3[1].image(blemish_steps["blemish_opened"], caption=f"10. Opening ({blemish_opening}x{blemish_opening})", use_container_width=True)
        rb3[2].image(blemish_steps["blemish_closed"], caption=f"11. Closing ({blemish_closing}x{blemish_closing})", use_container_width=True)
        rb3[3].image(blemish_steps["blemish_mask"], caption="12. Final blemish mask", use_container_width=True)

        st.caption(
            "The second blemish K-means clusters only Lab a,b. Lab L/brightness is not "
            "used for this grouping, so a normal shadow is less likely to become its own "
            "blemish region just because it is darker."
        )

    if is_mode_a:
        with st.expander("K-means cluster details"):
            df = pd.DataFrame(prepared["cluster_summary"])
            st.dataframe(df, use_container_width=True)
            st.caption("These clusters are used only to build the Mode A banana mask.")
    else:
        with st.expander("How Mode B works"):
            st.markdown(
                """
                **Mode B uses no K-means for banana segmentation.**

                1. Draw an ROI around the banana.
                2. Pixels outside the ROI start as definite background.
                3. Pixels inside the ROI start as possible foreground.
                4. GrabCut learns foreground/background colour models and refines the boundary.
                5. The largest connected foreground object is retained and holes are filled.
                6. Blemish detection runs inside the resulting banana mask and uses its own second-stage K-means.
                """
            )

    with st.expander("Blemish second-stage K-means cluster details"):
        blemish_df = pd.DataFrame(blemish_steps["blemish_cluster_summary"])
        st.dataframe(blemish_df, use_container_width=True)
        st.write(
            {
                "Healthy cluster IDs": blemish_steps["healthy_cluster_ids"],
                "Blemish cluster IDs": blemish_steps["blemish_cluster_ids"],
            }
        )
        st.caption(
            "Formal IoU/Dice evaluation should use one fixed ProcessingConfig for all "
            "test images. Do not tune these internal thresholds separately for each test image."
        )
