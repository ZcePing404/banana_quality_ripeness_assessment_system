from __future__ import annotations

from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image, ImageOps


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = MODULE_DIR / "models"


def find_ripeness_model_files(
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> tuple[Path | None, Path | None]:
    """Return the first H5 model and ``labels.txt`` found in ``model_dir``."""
    model_dir = Path(model_dir)

    preferred_names = [
        "keras_model.h5",
        "keras_Model.h5",
        "model.h5",
    ]

    model_path = None
    for name in preferred_names:
        candidate = model_dir / name
        if candidate.exists():
            model_path = candidate
            break

    if model_path is None:
        h5_files = sorted(model_dir.glob("*.h5")) if model_dir.exists() else []
        if h5_files:
            model_path = h5_files[0]

    labels_path = model_dir / "labels.txt"
    if not labels_path.exists():
        labels_path = None

    return model_path, labels_path


def load_labels(labels_path: str | Path) -> list[str]:
    """Load Teachable Machine labels and remove their numeric prefixes."""
    labels: list[str] = []

    with open(labels_path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue

            match = re.match(r"^\d+\s+(.*)$", line)
            labels.append(match.group(1).strip() if match else line)

    if not labels:
        raise ValueError("labels.txt does not contain any class labels.")

    return labels


def load_ripeness_model(
    model_path: str | Path,
    labels_path: str | Path,
):
    # Load the exported Teachable Machine H5 model with legacy TF-Keras."
    try:
        import tf_keras
    except ImportError as exc:
        raise ImportError(
            "The ripeness model requires the legacy TF-Keras package. "
            "Install it with: pip install tf-keras==2.20.1"
        ) from exc

    class CompatibleDepthwiseConv2D(tf_keras.layers.DepthwiseConv2D):
        def __init__(self, *args, groups=None, **kwargs):
            super().__init__(*args, **kwargs)

    model = tf_keras.models.load_model(
        str(model_path),
        compile=False,
        custom_objects={
            "DepthwiseConv2D": CompatibleDepthwiseConv2D,
        },
    )

    labels = load_labels(labels_path)
    return model, labels


def preprocess_bgr_image(image_bgr: np.ndarray) -> np.ndarray:
    """Prepare an OpenCV BGR image exactly as the TM image model expects."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Cannot classify an empty image.")

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb).convert("RGB")

    # Teachable Machine's image export uses a 224 x 224 centre crop/fit.
    image = ImageOps.fit(
        image,
        (224, 224),
        method=Image.Resampling.LANCZOS,
    )

    array = np.asarray(image, dtype=np.float32)
    array = (array / 127.5) - 1.0
    return np.expand_dims(array, axis=0)


def predict_ripeness(
    image_bgr: np.ndarray,
    model,
    labels: list[str],
) -> dict[str, object]:
    """Predict ripeness for one uploaded or camera-captured BGR image."""
    data = preprocess_bgr_image(image_bgr)
    output = model.predict(data, verbose=0)

    # Some Keras models return a list/tuple even for a single output.
    if isinstance(output, (list, tuple)):
        if not output:
            raise ValueError("The ripeness model returned no output.")
        output = output[0]

    scores = np.asarray(output, dtype=np.float32)
    if scores.ndim == 2:
        scores = scores[0]
    scores = scores.reshape(-1)

    if scores.size == 0:
        raise ValueError("The ripeness model returned an empty prediction.")

    if len(labels) != scores.size:
        raise ValueError(
            f"Model output has {scores.size} classes but labels.txt has "
            f"{len(labels)} labels."
        )

    best_index = int(np.argmax(scores))
    probabilities = {
        labels[index]: float(scores[index])
        for index in range(len(labels))
    }

    return {
        "class_name": labels[best_index],
        "confidence": float(scores[best_index]),
        "probabilities": probabilities,
    }
