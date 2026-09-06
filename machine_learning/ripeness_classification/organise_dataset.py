from pathlib import Path
import shutil

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

INPUT_ROOT = BASE_DIR / "data" / "raw"

OUTPUT_ROOT = BASE_DIR / "data" / "raw_by_class"

SPLITS = [
    "train",
    "valid",
    "test",
]

CLASSES = [
    "Unripe",
    "Ripe",
    "Overripe",
    "Rotten",
]

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}


def find_filename_column(df):
    """Find the column containing image filenames."""

    candidates = [
        "filename",
        "file",
        "image",
        "image_name",
        "name",
    ]

    lower_map = {
        str(column).lower().strip(): column
        for column in df.columns
    }

    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]

    # Roboflow CSVs commonly put filename first.
    return df.columns[0]


def determine_class(row, df_columns):
    """
    Support either:

    filename,class
    image.jpg,Ripe

    OR one-hot CSV:

    filename,Unripe,Ripe,Overripe,Rotten
    image.jpg,0,1,0,0
    """

    lower_map = {
        str(column).lower().strip(): column
        for column in df_columns
    }

    # --------------------------------------------------------
    # FORMAT 1:
    # class / label column
    # --------------------------------------------------------

    for possible_column in [
        "class",
        "label",
        "category",
    ]:
        if possible_column in lower_map:

            value = str(
                row[lower_map[possible_column]]
            ).strip()

            for class_name in CLASSES:
                if value.lower() == class_name.lower():
                    return class_name

            return None

    # --------------------------------------------------------
    # FORMAT 2:
    # one-hot class columns
    # --------------------------------------------------------

    detected_classes = []

    for class_name in CLASSES:

        matching_column = None

        for column in df_columns:
            if (
                str(column)
                .lower()
                .strip()
                == class_name.lower()
            ):
                matching_column = column
                break

        if matching_column is None:
            continue

        value = row[matching_column]

        try:
            value = float(value)
        except (TypeError, ValueError):
            continue

        if value > 0:
            detected_classes.append(
                class_name
            )

    if len(detected_classes) == 1:
        return detected_classes[0]

    return None


def find_image(
    split_dir,
    filename,
):
    """
    Locate the image referenced by _classes.csv.
    """

    filename = str(filename).strip()

    direct_path = split_dir / filename

    if direct_path.exists():
        return direct_path

    # Sometimes CSV filenames and actual files
    # differ slightly in extension/case.
    filename_stem = Path(filename).stem

    for path in split_dir.iterdir():

        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        if path.stem == filename_stem:
            return path

    return None


def organise_split(split):

    source_dir = (
        INPUT_ROOT
        / split
    )

    csv_path = (
        source_dir
        / "_classes.csv"
    )

    if not csv_path.exists():
        print(
            f"[ERROR] Missing CSV: "
            f"{csv_path}"
        )
        return

    print()
    print("=" * 60)
    print(
        f"Processing split: {split}"
    )
    print("=" * 60)

    df = pd.read_csv(
        csv_path
    )

    print(
        "CSV columns:",
        list(df.columns)
    )

    filename_column = (
        find_filename_column(df)
    )

    counts = {
        class_name: 0
        for class_name in CLASSES
    }

    skipped = 0
    missing = 0

    # Create folders first.
    for class_name in CLASSES:

        (
            OUTPUT_ROOT
            / split
            / class_name
        ).mkdir(
            parents=True,
            exist_ok=True
        )

    for _, row in df.iterrows():

        filename = row[
            filename_column
        ]

        class_name = determine_class(
            row,
            df.columns,
        )

        if class_name is None:

            print(
                f"[SKIP] Could not determine "
                f"class for: {filename}"
            )

            skipped += 1
            continue

        source_path = find_image(
            source_dir,
            filename,
        )

        if source_path is None:

            print(
                f"[MISSING] {filename}"
            )

            missing += 1
            continue

        destination = (
            OUTPUT_ROOT
            / split
            / class_name
            / source_path.name
        )

        # Copy, don't move.
        # Keep original dataset untouched.
        shutil.copy2(
            source_path,
            destination,
        )

        counts[class_name] += 1

    print()
    print("Class counts:")

    for class_name in CLASSES:
        print(
            f"  {class_name:10s}: "
            f"{counts[class_name]}"
        )

    print(
        f"Skipped       : {skipped}"
    )

    print(
        f"Missing files : {missing}"
    )


def main():

    for split in SPLITS:
        organise_split(split)

    print()
    print("=" * 60)
    print(
        "Dataset organisation completed."
    )
    print("=" * 60)


if __name__ == "__main__":
    main()