from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image, ImageOps

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import tensorflow as tf
import tf_keras


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# ------------------------------------------------------------
# CHANGE THESE TWO VALUES depending on which model you test
# ------------------------------------------------------------

MODEL_DIR = BASE_DIR / "models" / "baseline"

TEST_DIR = (
    BASE_DIR
    / "data"
    / "raw"
    / "test"
)

OUTPUT_DIR = (
    BASE_DIR
    / "evaluation"
    / "test_baseline"
)

MODEL_PATH = MODEL_DIR / "keras_model.h5"
LABELS_PATH = MODEL_DIR / "labels.txt"

IMAGE_SIZE = (224, 224)

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
}


# ============================================================
# TEACHABLE MACHINE MODEL COMPATIBILITY
# ============================================================

class CompatibleDepthwiseConv2D(
    tf_keras.layers.DepthwiseConv2D
):
    """
    Compatibility layer for some Teachable Machine
    exported H5 models.
    """

    def __init__(
        self,
        *args,
        groups=None,
        **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )


# ============================================================
# LOAD LABELS
# ============================================================

def load_labels(labels_path):
    """
    Teachable Machine labels may look like:

    0 Overripe
    1 Ripe
    2 Unripe

    This function removes the numeric prefix.
    """

    labels = []

    with open(
        labels_path,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            parts = line.split(
                " ",
                1
            )

            if (
                len(parts) == 2
                and parts[0].isdigit()
            ):
                label = parts[1]
            else:
                label = line

            labels.append(label)

    return labels


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_path):
    """
    Use the same preprocessing expected by
    the Teachable Machine model:

    RGB
        ↓
    224 × 224
        ↓
    float32
        ↓
    normalize to [-1, 1]
    """

    with Image.open(image_path) as image:

        image = image.convert("RGB")

        image = ImageOps.fit(
            image,
            IMAGE_SIZE,
            method=Image.Resampling.LANCZOS
        )

        image_array = np.asarray(
            image,
            dtype=np.float32
        )

    # Teachable Machine normalization
    image_array = (
        image_array / 127.5
    ) - 1.0

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("Loading model:")
    print(MODEL_PATH)

    model = tf_keras.models.load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={
            "DepthwiseConv2D":
                CompatibleDepthwiseConv2D
        }
    )

    return model


# ============================================================
# COLLECT VALIDATION IMAGES
# ============================================================

def collect_images(test_dir, class_names):

    records = []

    for class_name in class_names:

        class_dir = (
            test_dir
            / class_name
        )

        if not class_dir.exists():

            print(
                f"[WARNING] Missing class folder: "
                f"{class_dir}"
            )

            continue

        image_files = [
            path
            for path in class_dir.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower()
                in SUPPORTED_EXTENSIONS
            )
        ]

        print(
            f"{class_name}: "
            f"{len(image_files)} images"
        )

        for image_path in image_files:

            records.append({
                "path": image_path,
                "true_label": class_name
            })

    return records


# ============================================================
# RUN VALIDATION
# ============================================================

def evaluate():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    class_names = load_labels(
        LABELS_PATH
    )

    print()
    print("Classes:")
    print(class_names)
    print()

    model = load_model()

    records = collect_images(
        TEST_DIR,
        class_names
    )

    print()
    print(
        f"Total validation images: "
        f"{len(records)}"
    )
    print()

    results = []

    for index, record in enumerate(
        records,
        start=1
    ):

        image_path = record["path"]
        true_label = record["true_label"]

        try:

            data = preprocess_image(
                image_path
            )

            scores = model.predict(
                data,
                verbose=0
            )[0]

            predicted_index = int(
                np.argmax(scores)
            )

            predicted_label = (
                class_names[
                    predicted_index
                ]
            )

            confidence = float(
                scores[
                    predicted_index
                ]
            )

            correct = (
                predicted_label
                == true_label
            )

            result = {
                "filename":
                    image_path.name,

                "true_label":
                    true_label,

                "predicted_label":
                    predicted_label,

                "confidence":
                    confidence,

                "correct":
                    correct,
            }

            # Store probability/score
            # for every class
            for class_index, class_name in enumerate(
                class_names
            ):

                result[
                    f"score_{class_name}"
                ] = float(
                    scores[class_index]
                )

            results.append(result)

            status = (
                "CORRECT"
                if correct
                else "WRONG"
            )

            print(
                f"[{index}/{len(records)}] "
                f"{true_label:10s} -> "
                f"{predicted_label:10s} "
                f"{confidence * 100:6.2f}% "
                f"{status}"
            )

        except Exception as error:

            print(
                f"[ERROR] "
                f"{image_path}: "
                f"{error}"
            )

    # ========================================================
    # RESULTS DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        OUTPUT_DIR
        / "validation_predictions.csv",
        index=False
    )

    y_true = results_df[
        "true_label"
    ]

    y_pred = results_df[
        "predicted_label"
    ]

    # ========================================================
    # OVERALL METRICS
    # ========================================================

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    correct_count = int(
        results_df["correct"].sum()
    )

    total_count = len(
        results_df
    )

    wrong_count = (
        total_count
        - correct_count
    )

    summary_df = pd.DataFrame([
        {
            "accuracy":
                accuracy,

            "macro_precision":
                macro_precision,

            "macro_recall":
                macro_recall,

            "macro_f1":
                macro_f1,

            "correct":
                correct_count,

            "incorrect":
                wrong_count,

            "total":
                total_count,
        }
    ])

    summary_df.to_csv(
        OUTPUT_DIR
        / "validation_summary.csv",
        index=False
    )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    report = classification_report(
        y_true,
        y_pred,
        labels=class_names,
        output_dict=True,
        zero_division=0
    )

    report_df = (
        pd.DataFrame(report)
        .transpose()
    )

    report_df.to_csv(
        OUTPUT_DIR
        / "validation_classification_report.csv"
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=class_names
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names
    )

    display.plot(
        values_format="d"
    )

    plt.title(
        "Validation Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "validation_confusion_matrix.png",
        dpi=300
    )

    plt.close()

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    print(
        f"Accuracy        : "
        f"{accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Macro Precision : "
        f"{macro_precision:.4f}"
    )

    print(
        f"Macro Recall    : "
        f"{macro_recall:.4f}"
    )

    print(
        f"Macro F1-score  : "
        f"{macro_f1:.4f}"
    )

    print()
    print(
        f"Correct         : "
        f"{correct_count}/{total_count}"
    )

    print(
        f"Incorrect       : "
        f"{wrong_count}/{total_count}"
    )

    print()
    print("Classification Report:")
    print()

    print(
        classification_report(
            y_true,
            y_pred,
            labels=class_names,
            zero_division=0
        )
    )

    print()
    print(
        f"Results saved to:\n"
        f"{OUTPUT_DIR}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    evaluate()