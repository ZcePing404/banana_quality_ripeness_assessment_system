"""Flask entry point for banana surface-quality and ripeness assessment."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from banana_quality.config import ProcessingConfig
from banana_quality.pipeline import analyse_image
from banana_quality.visualisation import overlay_result
from flask_ui.ripeness import classify_ripeness


PROJECT_ROOT = Path(__file__).resolve().parent

app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "templates"),
    static_folder=str(PROJECT_ROOT / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_uploaded_image(raw_bytes: bytes) -> np.ndarray:
    if not raw_bytes:
        raise ValueError("No image data was received.")

    array = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)

    if image is None or image.size == 0:
        raise ValueError("The uploaded/captured file could not be decoded as an image.")

    return image


def _to_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)

    if array.dtype == np.uint8:
        return array

    if array.dtype == bool:
        return array.astype(np.uint8) * 255

    if np.issubdtype(array.dtype, np.floating):
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            return np.zeros(array.shape, dtype=np.uint8)

        minimum = float(finite.min())
        maximum = float(finite.max())

        if 0.0 <= minimum and maximum <= 1.0:
            array = array * 255.0
        elif maximum > minimum:
            array = (array - minimum) / (maximum - minimum) * 255.0
        else:
            array = np.zeros(array.shape, dtype=np.float32)

    return np.clip(array, 0, 255).astype(np.uint8)


def _encode_image_data_url(image: np.ndarray, *, png: bool = False) -> str:
    import base64

    array = _to_uint8(image)
    extension = ".png" if png else ".jpg"
    params = [] if png else [cv2.IMWRITE_JPEG_QUALITY, 92]

    ok, encoded = cv2.imencode(extension, array, params)
    if not ok:
        raise ValueError("Could not encode an output image for the web interface.")

    mime = "image/png" if png else "image/jpeg"
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def _as_float(form, name: str, default: float) -> float:
    return float(form.get(name, default))


def _as_int(form, name: str, default: int) -> int:
    return int(float(form.get(name, default)))


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _defaults() -> dict[str, int | float]:
    """Match the current Streamlit controls exactly."""
    config = ProcessingConfig()
    return {
        "kmeans_k": int(config.kmeans_k),
        "banana_score_threshold": float(config.banana_cluster_score_threshold),
        "banana_opening": int(config.banana_opening_kernel),
        "banana_closing": int(config.banana_closing_kernel),
        "blemish_k": int(config.blemish_kmeans_k),
        "blemish_chroma_threshold": float(config.blemish_cluster_chroma_threshold),
        "blemish_opening": int(config.blemish_opening_kernel),
        "blemish_closing": int(config.blemish_closing_kernel),
    }


def _processing_images(result: dict[str, object]) -> dict[str, object]:
    """Build diagnostic stages from the CURRENT processing pipeline."""
    prepared = result["prepared"]
    banana_steps = result["banana_steps"]
    blemish_steps = result["blemish_steps"]

    return {
        "banana": [
            {
                "caption": "1. Standardised image",
                "src": _encode_image_data_url(prepared["working_image"]),
            },
            {
                "caption": "2. Gaussian blur",
                "src": _encode_image_data_url(prepared["blurred"]),
            },
            {
                "caption": f"3. Whole-image K-means (K={result['kmeans_k']})",
                "src": _encode_image_data_url(prepared["kmeans_quantised_bgr"]),
            },
            {
                "caption": "4. Banana-colour prior",
                "src": _encode_image_data_url(prepared["banana_colour_prior"], png=True),
            },
            {
                "caption": "5. Cluster banana-likelihood",
                "src": _encode_image_data_url(prepared["banana_likelihood_map"], png=True),
            },
            {
                "caption": "6. Raw K-means banana candidate",
                "src": _encode_image_data_url(prepared["raw_banana_candidate"], png=True),
            },
            {
                "caption": (
                    f"7. Opening "
                    f"({result['banana_opening_kernel']}x{result['banana_opening_kernel']})"
                ),
                "src": _encode_image_data_url(banana_steps["banana_opened"], png=True),
            },
            {
                "caption": (
                    f"8. Closing "
                    f"({result['banana_closing_kernel']}x{result['banana_closing_kernel']})"
                ),
                "src": _encode_image_data_url(banana_steps["banana_closed"], png=True),
            },
        ],
        "blemish": [
            {
                "caption": "1. Area allowed for blemish analysis",
                "src": _encode_image_data_url(blemish_steps["inner_banana"], png=True),
            },
            {
                "caption": "2. Healthy colour-cluster family",
                "src": _encode_image_data_url(
                    blemish_steps["healthy_peel_candidates"], png=True
                ),
            },
            {
                "caption": "3. Second K-means using Lab a,b",
                "src": _encode_image_data_url(
                    blemish_steps["blemish_kmeans_quantised_bgr"]
                ),
            },
            {
                "caption": "4. a,b distance from healthy clusters",
                "src": _encode_image_data_url(
                    blemish_steps["blemish_cluster_distance_map"], png=True
                ),
            },
            {
                "caption": "5. Region blemish candidate",
                "src": _encode_image_data_url(
                    blemish_steps["kmeans_region_candidate"], png=True
                ),
            },
            {
                "caption": "6. Bright-highlight exclusion",
                "src": _encode_image_data_url(
                    blemish_steps["bright_highlight_mask"], png=True
                ),
            },
            {
                "caption": "7. Raw blemish candidate",
                "src": _encode_image_data_url(
                    blemish_steps["raw_blemish_candidate"], png=True
                ),
            },
            {
                "caption": (
                    f"8. Opening "
                    f"({result['blemish_opening_kernel']}x{result['blemish_opening_kernel']})"
                ),
                "src": _encode_image_data_url(
                    blemish_steps["blemish_opened"], png=True
                ),
            },
            {
                "caption": (
                    f"9. Closing "
                    f"({result['blemish_closing_kernel']}x{result['blemish_closing_kernel']})"
                ),
                "src": _encode_image_data_url(
                    blemish_steps["blemish_closed"], png=True
                ),
            },
            {
                "caption": "10. Final blemish mask",
                "src": _encode_image_data_url(
                    blemish_steps["blemish_mask"], png=True
                ),
            },
        ],
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return render_template("index.html", defaults=_defaults())


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/api/analyse")
def analyse():
    try:
        image_file = request.files.get("image")
        if image_file is None:
            return jsonify(
                {"ok": False, "error": "Choose or capture an image first."}
            ), 400

        image = _decode_uploaded_image(image_file.read())

        defaults = _defaults()
        settings = {
            "kmeans_k": _as_int(
                request.form, "kmeans_k", defaults["kmeans_k"]
            ),
            "banana_score_threshold": _as_float(
                request.form,
                "banana_score_threshold",
                defaults["banana_score_threshold"],
            ),
            "banana_opening": _as_int(
                request.form, "banana_opening", defaults["banana_opening"]
            ),
            "banana_closing": _as_int(
                request.form, "banana_closing", defaults["banana_closing"]
            ),
            "blemish_k": _as_int(
                request.form, "blemish_k", defaults["blemish_k"]
            ),
            "blemish_chroma_threshold": _as_float(
                request.form,
                "blemish_chroma_threshold",
                defaults["blemish_chroma_threshold"],
            ),
            "blemish_opening": _as_int(
                request.form, "blemish_opening", defaults["blemish_opening"]
            ),
            "blemish_closing": _as_int(
                request.form, "blemish_closing", defaults["blemish_closing"]
            ),
        }

        # Match the current Streamlit app.py exactly.
        config = replace(
            ProcessingConfig(),
            banana_cluster_score_threshold=float(
                settings["banana_score_threshold"]
            ),
            blemish_kmeans_k=int(settings["blemish_k"]),
            blemish_cluster_chroma_threshold=float(
                settings["blemish_chroma_threshold"]
            ),
        )

        result = analyse_image(
            image,
            kmeans_k=int(settings["kmeans_k"]),
            banana_opening_kernel=int(settings["banana_opening"]),
            banana_closing_kernel=int(settings["banana_closing"]),
            blemish_opening_kernel=int(settings["blemish_opening"]),
            blemish_closing_kernel=int(settings["blemish_closing"]),
            config=config,
        )

        # IMPORTANT: match current Streamlit behavior.
        # Reuse the same final banana mask for ripeness classification.
        ripeness_result = classify_ripeness(
            image,
            result["banana_mask"],
        )

        overlay = overlay_result(
            image,
            result["banana_mask"],
            result["blemish_mask"],
        )

        images = {
            "original": _encode_image_data_url(image),
            "banana_mask": _encode_image_data_url(
                result["banana_mask"], png=True
            ),
            "blemish_mask": _encode_image_data_url(
                result["blemish_mask"], png=True
            ),
            "overlay": _encode_image_data_url(overlay),
        }

        if ripeness_result.get("available"):
            segmented = ripeness_result.get("segmented_image")
            ripeness_mask = ripeness_result.get("banana_mask")

            if segmented is not None:
                images["ripeness_segmented"] = _encode_image_data_url(segmented)

            if ripeness_mask is not None:
                images["ripeness_mask"] = _encode_image_data_url(
                    ripeness_mask, png=True
                )

        prepared = result["prepared"]
        blemish_steps = result["blemish_steps"]

        payload = {
            "ok": True,
            "metrics": {
                "ripeness": (
                    str(ripeness_result.get("class_name", "Unknown"))
                    if ripeness_result.get("available")
                    else "Unavailable"
                ),
                "ripeness_confidence": (
                    float(ripeness_result.get("confidence", 0.0))
                    if ripeness_result.get("available")
                    else None
                ),
                "blemish_percentage": float(result["blemish_percentage"]),
                "surface_quality": str(result["quality_grade"]),
            },
            "ripeness": {
                "available": bool(ripeness_result.get("available")),
                "error": ripeness_result.get("error"),
                "probabilities": _json_safe(
                    ripeness_result.get("probabilities", {})
                ),
            },
            "segmentation_caption": (
                "Shared banana segmentation: automatic K-means; selected clusters: "
                f"{prepared['selected_clusters']}"
            ),
            "images": images,
            "processing": _processing_images(result),
            "cluster_summary": _json_safe(
                prepared.get("cluster_summary", [])
            ),
            "blemish_cluster_summary": _json_safe(
                blemish_steps.get("blemish_cluster_summary", [])
            ),
            "healthy_cluster_ids": _json_safe(
                blemish_steps.get("healthy_cluster_ids", [])
            ),
            "blemish_cluster_ids": _json_safe(
                blemish_steps.get("blemish_cluster_ids", [])
            ),
        }

        return jsonify(payload)

    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False,
    )
