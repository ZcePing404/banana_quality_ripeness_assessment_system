from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import base64
import json
from io import BytesIO
from flask import send_file
from banana_quality.calibration import calculate_pixels_per_cm_from_line, measure_banana_from_mask, draw_measurement_box

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
    array = _to_uint8(image)
    extension = ".png" if png else ".jpg"
    params = [] if png else [cv2.IMWRITE_JPEG_QUALITY, 90]

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


def _as_bool(form, name: str, default: bool = True) -> bool:
    value = str(form.get(name, "1" if default else "0")).strip().lower()
    return value not in {"0", "false", "no", "off"}


def _ripeness_method(form) -> str:
    """Return the selected ripeness-classification pipeline."""
    method = str(form.get("ripeness_method", "baseline")).strip().lower()

    if method not in {"baseline", "segmented"}:
        raise ValueError(
            "Ripeness method must be either 'baseline' or 'segmented'."
        )

    return method


def _defaults() -> dict[str, int | float]:
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


def _compact_processing_images(
    result: dict[str, object],
    overlay: np.ndarray,
) -> dict[str, list[dict[str, str]]]:

    prepared = result["prepared"]
    blemish_steps = result["blemish_steps"]

    return {
        "preprocessing": [
            {"caption": caption, "src": _encode_image_data_url(prepared[key])}
            for caption, key in [("Aspect-ratio resize", "working_image"), ("CLAHE enhancement", "clahe_enhanced"), ("Gaussian filtering", "blurred")]
        ],
        "banana": [
            {
                "caption": "Input",
                "src": _encode_image_data_url(prepared["working_image"]),
            },
            {
                "caption": "K-means candidate",
                "src": _encode_image_data_url(prepared["raw_banana_candidate"], png=True),
            },
            {
                "caption": "Final mask",
                "src": _encode_image_data_url(result["banana_mask"], png=True),
            },
        ],
        "blemish": [
            {
                "caption": "Banana mask",
                "src": _encode_image_data_url(result["banana_mask"], png=True),
            },
            {
                "caption": "K-means candidate",
                "src": _encode_image_data_url(
                    blemish_steps["kmeans_region_candidate"], png=True
                ),
            },
            {
                "caption": "Final mask",
                "src": _encode_image_data_url(result["blemish_mask"], png=True),
            },
            {
                "caption": "Overlay",
                "src": _encode_image_data_url(overlay),
            },
        ],
    }

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
            return jsonify({"ok": False, "error": "Choose or capture an image first."}), 400

        image = _decode_uploaded_image(image_file.read())
        include_ripeness = _as_bool(request.form, "include_ripeness", True)
        ripeness_method = _ripeness_method(request.form)
        compact_response = _as_bool(request.form, "compact_response", False)

        defaults = _defaults()
        settings = {
            "kmeans_k": _as_int(request.form, "kmeans_k", defaults["kmeans_k"]),
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
            "blemish_k": _as_int(request.form, "blemish_k", defaults["blemish_k"]),
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

        config = replace(
            ProcessingConfig(),
            banana_cluster_score_threshold=float(settings["banana_score_threshold"]),
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

        overlay = overlay_result(
            image,
            result["banana_mask"],
            result["blemish_mask"],
        )

        calibration = {"applied": False}
        if _as_bool(request.form, "calibration_enabled", False):
            points = np.asarray(json.loads(request.form.get("calibration_points", "[]")), dtype=float)
            if points.shape != (2, 2) or not np.isfinite(points).all():
                raise ValueError("Select two valid reference points.")
            if (points < 0).any() or (points[:, 0] >= image.shape[1]).any() or (points[:, 1] >= image.shape[0]).any():
                raise ValueError("Reference points must lie inside this image.")
            scale = calculate_pixels_per_cm_from_line(points, float(request.form.get("reference_size_cm", 0)))
            measurement = measure_banana_from_mask(result["banana_mask"], scale)
            overlay = draw_measurement_box(overlay, measurement)
            calibration = {"applied": True, **{key: float(measurement[key]) for key in ("length_cm", "width_cm", "pixels_per_cm")}}

        images = {
            "overlay": _encode_image_data_url(overlay),
        }
        if not compact_response:
            images.update({
                "banana_mask": _encode_image_data_url(result["banana_mask"], png=True),
                "blemish_mask": _encode_image_data_url(result["blemish_mask"], png=True),
            })

        ripeness_payload: dict[str, object]
        ripeness_name: str | None = None
        ripeness_confidence: float | None = None

        if include_ripeness:
            ripeness_result = classify_ripeness(
                image,
                banana_mask=result["banana_mask"],
                method=ripeness_method,
            )

            ripeness_name = (
                str(ripeness_result.get("class_name", "Unknown"))
                if ripeness_result.get("available")
                else "Unavailable"
            )
            ripeness_confidence = (
                float(ripeness_result.get("confidence", 0.0))
                if ripeness_result.get("available")
                else None
            )

            classifier_input = ripeness_result.get("classifier_input")
            if classifier_input is not None and not compact_response:
                images["ripeness_input"] = _encode_image_data_url(classifier_input)

            ripeness_payload = {
                "skipped": False,
                "available": bool(ripeness_result.get("available")),
                "error": ripeness_result.get("error"),
                "method": str(ripeness_result.get("method", ripeness_method)),
                "method_label": str(
                    ripeness_result.get(
                        "method_label",
                        "Baseline model"
                        if ripeness_method == "baseline"
                        else "Segmented model",
                    )
                ),
                "input_caption": str(
                    ripeness_result.get(
                        "input_caption",
                        "Original / cropped classifier input"
                        if ripeness_method == "baseline"
                        else "Background-removed classifier input",
                    )
                ),
                "probabilities": {
                    str(name): float(value)
                    for name, value in ripeness_result.get("probabilities", {}).items()
                },
            }
        else:
            ripeness_payload = {
                "skipped": True,
                "available": None,
                "error": None,
                "method": ripeness_method,
                "method_label": (
                    "Baseline model"
                    if ripeness_method == "baseline"
                    else "Segmented model"
                ),
                "input_caption": None,
                "probabilities": {},
            }

        payload = {
            "ok": True,
            "calibration": calibration,
            "settings": settings,
            "metrics": {
                "ripeness": ripeness_name,
                "ripeness_confidence": ripeness_confidence,
                "blemish_percentage": float(result["blemish_percentage"]),
                "surface_quality": str(result["quality_grade"]),
            },
            "ripeness": ripeness_payload,
            "images": images,
            "processing": (
                {"banana": [], "blemish": []}
                if compact_response
                else _compact_processing_images(result, overlay)
            ),
            "filename": image_file.filename or "image",
            "compact": compact_response,
        }

        return jsonify(payload)

    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Image analysis failed")
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.errorhandler(413)
def too_large(error):
    return jsonify(ok=False, error="Image exceeds the 16 MB upload limit. Resize or compress it and retry."), 413

@app.post("/api/report")
def report():
    from flask_ui.reporting import build_report
    body = request.get_json(silent=True) or {}
    results = body.get("results", [])
    if not isinstance(results, list) or not results or len(results) > 200:
        return jsonify(ok=False, error="Supply between 1 and 200 results."), 400
    try:
        return send_file(BytesIO(build_report(results)), mimetype="application/pdf", as_attachment=True, download_name="banana_assessment_report.pdf")
    except (ValueError, TypeError, KeyError) as exc:
        return jsonify(ok=False, error=str(exc)), 400

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False,
    )
