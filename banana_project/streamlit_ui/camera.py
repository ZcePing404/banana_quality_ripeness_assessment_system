"""Local USB camera support for the Streamlit interface."""

from __future__ import annotations

import os

import cv2
import streamlit as st

def open_usb_camera(camera_index: int):
    """Open a USB/local camera.

    On Windows, DirectShow is tried first because it is often more reliable
    for external USB webcams. If that fails, OpenCV's default backend is used.
    """
    if os.name == "nt":
        cap = cv2.VideoCapture(int(camera_index), cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()

    return cv2.VideoCapture(int(camera_index))


def release_usb_camera() -> None:
    """Release the currently opened USB camera, if any."""
    cap = st.session_state.get("_usb_camera_cap")
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass

    st.session_state.pop("_usb_camera_cap", None)
    st.session_state.pop("_usb_camera_index", None)
    st.session_state.pop("usb_latest_frame", None)


def ensure_usb_camera(camera_index: int):
    """Keep one OpenCV camera open across Streamlit fragment refreshes."""
    old_index = st.session_state.get("_usb_camera_index")
    cap = st.session_state.get("_usb_camera_cap")

    # Re-open only when the device changes or the old camera is no longer valid.
    if (
        cap is None
        or old_index != int(camera_index)
        or not cap.isOpened()
    ):
        release_usb_camera()
        cap = open_usb_camera(int(camera_index))

        if cap.isOpened():
            # These are requests only; the webcam may choose the nearest supported mode.
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            st.session_state["_usb_camera_cap"] = cap
            st.session_state["_usb_camera_index"] = int(camera_index)

    return cap


@st.fragment(run_every=0.10)
def show_usb_live_stream(camera_index: int) -> None:
    """Continuously read and display frames from the selected local USB camera.

    The fragment refreshes only this camera section about 10 times per second,
    instead of repeatedly rerunning the whole image-processing application.
    """
    cap = ensure_usb_camera(camera_index)

    if cap is None or not cap.isOpened():
        st.error(
            f"Cannot open camera index {camera_index}. "
            "Try another index or close other programmes using the camera."
        )
        return

    ok, frame = cap.read()
    if not ok or frame is None:
        st.warning("Camera opened, but a video frame could not be read.")
        return

    # Keep the newest frame so the normal Streamlit Capture button can freeze it.
    st.session_state["usb_latest_frame"] = frame.copy()

    st.image(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        caption=f"Live USB camera preview — device {camera_index}",
        use_container_width=True,
    )

