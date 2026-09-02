from pathlib import Path

import cv2

from segmentation import segment_banana


# ============================================================
# DATASET CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_ROOT = BASE_DIR / "data" / "raw"
OUTPUT_ROOT = BASE_DIR / "data" / "segmented"
MASK_ROOT = BASE_DIR / "data" / "segmentation_masks"

SPLITS = [
    "train",
    "valid",
    "test",
]

CLASSES = [
    "Unripe",
    "Ripe",
    "Overripe",
]

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}
# ============================================================
# PROCESS SINGLE IMAGE
# ============================================================

def process_image(input_path, output_path, mask_path):

    image = cv2.imread(str(input_path))

    if image is None:
        print(f"Could not read: {input_path}")
        return False

    segmented, mask = segment_banana(image)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    mask_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.imwrite(
        str(output_path),
        segmented
    )

    cv2.imwrite(
        str(mask_path),
        mask
    )

    return True


# ============================================================
# PROCESS DATASET
# ============================================================

def process_dataset():

    total = 0
    successful = 0
    failed = 0

    print("=" * 70)
    print("BANANA DATASET SEGMENTATION")
    print("=" * 70)

    print(f"Input:  {INPUT_ROOT}")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"Masks:  {MASK_ROOT}")
    print()

    if not INPUT_ROOT.exists():
        raise FileNotFoundError(
            f"Input directory does not exist:\n"
            f"{INPUT_ROOT}"
        )

    for split in SPLITS:

        for class_name in CLASSES:

            source_dir = (
                INPUT_ROOT
                / split
                / class_name
            )

            if not source_dir.exists():

                print(
                    f"[WARNING] Folder not found: "
                    f"{source_dir}"
                )

                continue

            image_files = [
                path
                for path in source_dir.rglob("*")
                if (
                    path.is_file()
                    and path.suffix.lower()
                    in SUPPORTED_EXTENSIONS
                )
            ]

            print(
                f"\n{split}/{class_name}: "
                f"{len(image_files)} images"
            )

            for image_path in image_files:

                total += 1

                relative_path = (
                    image_path.relative_to(
                        INPUT_ROOT
                    )
                )

                output_path = (
                    OUTPUT_ROOT
                    / relative_path
                )

                mask_path = (
                    MASK_ROOT
                    / relative_path
                )

                success = process_image(
                    image_path,
                    output_path,
                    mask_path
                )

                if success:
                    successful += 1

                    print(
                        f"  [{successful}] "
                        f"{relative_path}"
                    )

                else:
                    failed += 1

    print()
    print("=" * 70)
    print("SEGMENTATION COMPLETE")
    print("=" * 70)

    print(f"Total images : {total}")
    print(f"Successful   : {successful}")
    print(f"Failed       : {failed}")

    print()
    print(
        "Segmented images saved to:"
    )
    print(OUTPUT_ROOT)

    print()
    print(
        "Binary masks saved to:"
    )
    print(MASK_ROOT)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    process_dataset()