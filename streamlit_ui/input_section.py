"""Image-source section for uploads and local USB camera capture."""

from __future__ import annotations

import cv2
import streamlit as st

from .camera import release_usb_camera, show_usb_live_stream
from .image_io import decode_image


def render_image_input():
    """Render the original image input section and return (image, input_source)."""
    st.subheader("Image input")
    input_source = st.radio(
        "Choose input source",
        ["Upload image", "USB camera"],
        horizontal=True,
    )

    image = None

    if input_source == "Upload image":
        # Release the webcam when it is not being used.
        if "_usb_camera_cap" in st.session_state:
            release_usb_camera()

        uploaded = st.file_uploader(
            "Choose one banana image",
            type=["jpg", "jpeg", "png"],
        )
        image = decode_image(uploaded)

    else:
        st.caption(
            "This is a real-time local USB-camera preview. The camera must be connected "
            "to the same computer that is running Streamlit."
        )

        st.info(
            "Capture setup for more reliable blemish/shadow separation: use a plain "
            "background, keep the camera position fixed, and use soft/diffuse light from "
            "both sides of the banana. Avoid one strong directional lamp or direct glare."
        )

        camera_index = st.selectbox(
            "USB camera device index",
            options=[0, 1, 2, 3, 4, 5],
            index=1,
            help=(
                "Usually 0 is the built-in laptop camera and 1 is the first USB webcam, "
                "but the numbering depends on the computer."
            ),
        )

        # This fragment refreshes independently, producing the live video preview.
        show_usb_live_stream(int(camera_index))

        capture_col, clear_col = st.columns(2)

        with capture_col:
            if st.button(
                "Capture current frame",
                type="primary",
                use_container_width=True,
            ):
                latest = st.session_state.get("usb_latest_frame")

                if latest is None:
                    st.error(
                        "No live frame is available yet. Wait for the preview to appear "
                        "and then press Capture current frame."
                    )
                else:
                    # Freeze a copy. The analysis/ROI below uses this captured image,
                    # while the live camera preview can continue running above.
                    st.session_state["usb_captured_frame"] = latest.copy()
                    st.session_state["usb_captured_index"] = int(camera_index)

        with clear_col:
            if st.button("Clear captured image", use_container_width=True):
                st.session_state.pop("usb_captured_frame", None)
                st.session_state.pop("usb_captured_index", None)

        if "usb_captured_frame" in st.session_state:
            image = st.session_state["usb_captured_frame"]
            used_index = st.session_state.get("usb_captured_index", "?")

            st.success(
                "Frame captured. This frozen image is now used for banana/blemish analysis."
            )
            st.image(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                caption=f"Captured frame used for analysis — device {used_index}",
                use_container_width=True,
            )

    return image, input_source
