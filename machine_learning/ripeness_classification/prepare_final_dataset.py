from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from hashlib import sha256
from pathlib import Path
import csv
import shutil

import numpy as np
from PIL import Image, ImageOps, ImageDraw
import imagehash

# CONFIGURATION

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"

RAW_ROOT = DATA_DIR / "raw"
CLASS_ROOT = DATA_DIR / "raw_by_class"

SPLITS = ["train", "valid", "test"]
CLASSES = ["Unripe", "Ripe", "Overripe", "Rotten"]

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

STRICT_PHASH_THRESHOLD = 2
REVIEW_PHASH_THRESHOLD = 5

# You previously reviewed the near-duplicate pairs and confirmed they
# appeared to be the same source images.
AUTO_CLEAN_REVIEW_NEAR_DUPLICATES = True

# Lower number = higher priority = preserve.
SPLIT_PRIORITY = {
    "train": 0,
    "valid": 1,
    "test": 2,
}

BACKUP_EXISTING_RAW_BY_CLASS = True

AUDIT_DIR = DATA_DIR / "final_leakage_audit"
BEFORE_CSV = AUDIT_DIR / "cross_split_duplicates_before_cleaning.csv"
AFTER_CSV = AUDIT_DIR / "cross_split_duplicates_after_cleaning.csv"
REVIEW_DIR = AUDIT_DIR / "review_pairs_before_cleaning"

QUARANTINE_DIR = CLASS_ROOT / "_removed_cross_split_duplicates"

CLASS_DISTRIBUTION_CSV = CLASS_ROOT / "class_distribution_final.csv"
ORGANISATION_LOG_CSV = CLASS_ROOT / "organisation_log.csv"
REMOVAL_LOG_CSV = CLASS_ROOT / "duplicate_removal_log.csv"


# ============================================================
# BASIC HELPERS
# ============================================================

def normalise_key(value: str) -> str:
    return "".join(
        ch.lower()
        for ch in str(value)
        if ch.isalnum()
    )


CLASS_ALIASES = {
    "unripe": "Unripe",
    "ripe": "Ripe",
    "overripe": "Overripe",
    "rotten": "Rotten",
}


def normalise_class_name(value: str) -> str | None:
    return CLASS_ALIASES.get(normalise_key(value))


def truthy_label_value(value: str) -> bool:
    text = str(value).strip().lower()

    if text in {"1", "true", "yes", "y", "x"}:
        return True

    try:
        return float(text) > 0.5
    except (TypeError, ValueError):
        return False


def validate_raw_structure() -> None:
    if not RAW_ROOT.exists():
        raise FileNotFoundError(
            f"Raw dataset folder not found:\n{RAW_ROOT}"
        )

    missing = [
        split
        for split in SPLITS
        if not (RAW_ROOT / split).exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing raw split folder(s): "
            + ", ".join(missing)
        )


def prepare_output() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    if REVIEW_DIR.exists():
        shutil.rmtree(REVIEW_DIR)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    if CLASS_ROOT.exists() and any(CLASS_ROOT.iterdir()):
        if BACKUP_EXISTING_RAW_BY_CLASS:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = DATA_DIR / f"raw_by_class_backup_{stamp}"
            print(f"[BACKUP] Existing raw_by_class -> {backup}")
            shutil.move(str(CLASS_ROOT), str(backup))
        else:
            shutil.rmtree(CLASS_ROOT)

    CLASS_ROOT.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        for class_name in CLASSES:
            (CLASS_ROOT / split / class_name).mkdir(
                parents=True,
                exist_ok=True,
            )


# ============================================================
# STAGE 1 — READ *_classes.csv AND ORGANISE BY CLASS
# ============================================================

def locate_classes_csv(split_dir: Path) -> Path:
    candidates = sorted(
        path
        for path in split_dir.glob("*.csv")
        if "classes" in path.name.lower()
    )

    preferred = {"_classes.csv", "classes.csv"}

    for path in candidates:
        if path.name.lower() in preferred:
            return path

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise FileNotFoundError(
            f"No classes CSV found in:\n{split_dir}\n"
            "Expected something like _classes.csv."
        )

    raise RuntimeError(
        "Multiple possible classes CSV files found in:\n"
        f"{split_dir}\n"
        + "\n".join(str(p) for p in candidates)
    )


def infer_filename_column(fieldnames: list[str]) -> str:
    candidates = {
        "filename",
        "file",
        "image",
        "imagename",
        "imagefile",
        "filepath",
        "path",
    }

    for field in fieldnames:
        if normalise_key(field) in candidates:
            return field

    if fieldnames:
        return fieldnames[0]

    raise ValueError("Classes CSV has no header columns.")


def infer_direct_label_column(fieldnames: list[str]) -> str | None:
    candidates = {
        "class",
        "label",
        "classname",
        "category",
        "ripeness",
        "ripenessclass",
    }

    for field in fieldnames:
        if normalise_key(field) in candidates:
            return field

    return None


def read_labels(split: str) -> tuple[dict[str, str], Path]:
    split_dir = RAW_ROOT / split
    csv_path = locate_classes_csv(split_dir)

    labels: dict[str, str] = {}

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(
                f"No header found in {csv_path}"
            )

        fields = list(reader.fieldnames)
        filename_column = infer_filename_column(fields)
        direct_label_column = infer_direct_label_column(fields)

        class_columns = []
        for field in fields:
            class_name = normalise_class_name(field)
            if class_name is not None:
                class_columns.append((field, class_name))

        if direct_label_column is None and not class_columns:
            raise ValueError(
                f"Could not infer classes from {csv_path}\n"
                f"Columns found: {fields}"
            )

        for row_number, row in enumerate(reader, start=2):
            raw_filename = str(
                row.get(filename_column, "")
            ).strip()

            if not raw_filename:
                continue

            filename_key = Path(raw_filename).name.lower()
            class_name = None

            if direct_label_column is not None:
                class_name = normalise_class_name(
                    row.get(direct_label_column, "")
                )

            if class_name is None and class_columns:
                positive_classes = [
                    canonical
                    for column, canonical in class_columns
                    if truthy_label_value(row.get(column, ""))
                ]

                positive_classes = list(
                    dict.fromkeys(positive_classes)
                )

                if len(positive_classes) == 1:
                    class_name = positive_classes[0]

                elif len(positive_classes) > 1:
                    raise ValueError(
                        f"Multiple positive classes in {csv_path}, "
                        f"row {row_number}, image {raw_filename}: "
                        f"{positive_classes}"
                    )

            if class_name is None:
                print(
                    f"[WARNING] Could not determine class for "
                    f"{split}/{raw_filename}; skipped."
                )
                continue

            labels[filename_key] = class_name

    return labels, csv_path


def build_raw_image_lookup(split: str) -> dict[str, Path]:
    lookup: dict[str, Path] = {}

    for path in (RAW_ROOT / split).rglob("*"):
        if (
            path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ):
            key = path.name.lower()

            if key in lookup:
                raise RuntimeError(
                    "Duplicate filename within the same raw split:\n"
                    f"  {lookup[key]}\n"
                    f"  {path}\n\n"
                    "The classes CSV identifies images by filename, "
                    "so this is ambiguous."
                )

            lookup[key] = path.resolve()

    return lookup


def organise_by_class() -> None:
    print()
    print("=" * 78)
    print("STAGE 1 — CREATE FRESH raw_by_class FROM *_classes.csv")
    print("=" * 78)

    log_rows = []

    for split in SPLITS:
        labels, csv_path = read_labels(split)
        lookup = build_raw_image_lookup(split)

        copied = 0
        missing_from_disk = 0
        unlabelled_on_disk = 0

        print()
        print(f"[{split.upper()}]")
        print(f"Classes CSV : {csv_path}")
        print(f"Labels read : {len(labels)}")
        print(f"Images found: {len(lookup)}")

        for filename_key, class_name in labels.items():
            source = lookup.get(filename_key)

            if source is None:
                missing_from_disk += 1
                log_rows.append({
                    "split": split,
                    "filename": filename_key,
                    "class": class_name,
                    "status": "CSV_IMAGE_NOT_FOUND",
                    "source": "",
                    "destination": "",
                })
                continue

            destination = (
                CLASS_ROOT
                / split
                / class_name
                / source.name
            )

            shutil.copy2(source, destination)
            copied += 1

            log_rows.append({
                "split": split,
                "filename": source.name,
                "class": class_name,
                "status": "COPIED",
                "source": str(source),
                "destination": str(destination),
            })

        for filename_key, source in lookup.items():
            if filename_key not in labels:
                unlabelled_on_disk += 1
                log_rows.append({
                    "split": split,
                    "filename": source.name,
                    "class": "",
                    "status": "IMAGE_NOT_LISTED_IN_CLASSES_CSV",
                    "source": str(source),
                    "destination": "",
                })

        print(f"Copied            : {copied}")
        print(f"CSV images missing: {missing_from_disk}")
        print(f"Images unlabelled : {unlabelled_on_disk}")

    with ORGANISATION_LOG_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        fieldnames = [
            "split",
            "filename",
            "class",
            "status",
            "source",
            "destination",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_rows)


# ============================================================
# DUPLICATE-AUDIT HELPERS
# ============================================================

def collect_classified_images() -> list[dict]:
    items = []

    for split in SPLITS:
        split_dir = CLASS_ROOT / split

        for class_name in CLASSES:
            class_dir = split_dir / class_name

            for path in class_dir.rglob("*"):
                if (
                    path.is_file()
                    and path.suffix.lower() in SUPPORTED_EXTENSIONS
                ):
                    items.append({
                        "split": split,
                        "class": class_name,
                        "path": path.resolve(),
                    })

    return items


def file_sha256(path: Path) -> str:
    hasher = sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            hasher.update(chunk)

    return hasher.hexdigest()


def decoded_pixel_hash(path: Path) -> str:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(
            image
        ).convert("RGB")

        hasher = sha256()
        hasher.update(
            str(image.size).encode("utf-8")
        )
        hasher.update(image.tobytes())

        return hasher.hexdigest()


def perceptual_hash(path: Path):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(
            image
        ).convert("RGB")
        return imagehash.phash(
            image,
            hash_size=8,
        )


def resized_mae(
    path_a: Path,
    path_b: Path,
    size=(256, 256),
) -> float:

    def load(path: Path) -> np.ndarray:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(
                image
            ).convert("RGB")
            image = image.resize(
                size,
                Image.Resampling.LANCZOS,
            )
            return np.asarray(
                image,
                dtype=np.float32,
            )

    a = load(path_a)
    b = load(path_b)

    return float(
        np.mean(
            np.abs(a - b)
        )
    )


def fit_for_sheet(
    path: Path,
    target=(420, 320),
) -> Image.Image:

    with Image.open(path) as image:
        image = ImageOps.exif_transpose(
            image
        ).convert("RGB")
        image.thumbnail(
            target,
            Image.Resampling.LANCZOS,
        )

        canvas = Image.new(
            "RGB",
            target,
            "white",
        )

        x = (target[0] - image.width) // 2
        y = (target[1] - image.height) // 2
        canvas.paste(image, (x, y))

        return canvas


def create_review_pair(
    number: int,
    row: dict,
) -> Path:

    left = fit_for_sheet(
        Path(row["path_a"])
    )
    right = fit_for_sheet(
        Path(row["path_b"])
    )

    width = left.width + right.width
    height = left.height + 90

    sheet = Image.new(
        "RGB",
        (width, height),
        "white",
    )

    sheet.paste(left, (0, 62))
    sheet.paste(right, (left.width, 62))

    draw = ImageDraw.Draw(sheet)

    draw.text(
        (10, 8),
        (
            f"{row['match_type']} | "
            f"pHash={row['phash_distance']} | "
            f"MAE={row['resized_mae']:.2f}"
        ),
        fill="black",
    )

    draw.text(
        (10, 30),
        (
            f"{row['split_a']}/{row['class_a']}: "
            f"{Path(row['path_a']).name}"
        ),
        fill="black",
    )

    draw.text(
        (left.width + 10, 30),
        (
            f"{row['split_b']}/{row['class_b']}: "
            f"{Path(row['path_b']).name}"
        ),
        fill="black",
    )

    output = (
        REVIEW_DIR
        / f"pair_{number:04d}.jpg"
    )

    sheet.save(
        output,
        quality=92,
    )

    return output


# ============================================================
# STAGE 2 — AUDIT raw_by_class
# ============================================================

def audit_classified_dataset(
    *,
    output_csv: Path,
    create_review_images: bool,
) -> list[dict]:

    items = collect_classified_images()

    print(f"Images being audited: {len(items)}")

    valid_items = []

    for index, item in enumerate(
        items,
        start=1,
    ):
        path = item["path"]

        print(
            f"[HASH {index}/{len(items)}] "
            f"{item['split']}/{item['class']}: "
            f"{path.name}"
        )

        try:
            item["file_hash"] = file_sha256(path)
            item["pixel_hash"] = decoded_pixel_hash(path)
            item["phash"] = perceptual_hash(path)
            valid_items.append(item)

        except Exception as exc:
            print(
                f"    [SKIP unreadable image] {exc}"
            )

    results = []
    checked_pairs = 0

    for i in range(len(valid_items)):
        a = valid_items[i]

        for j in range(
            i + 1,
            len(valid_items),
        ):
            b = valid_items[j]

            if a["split"] == b["split"]:
                continue

            checked_pairs += 1

            phash_distance = int(
                a["phash"] - b["phash"]
            )

            if a["file_hash"] == b["file_hash"]:
                match_type = "EXACT_FILE"

            elif a["pixel_hash"] == b["pixel_hash"]:
                match_type = "EXACT_PIXELS"

            elif phash_distance <= STRICT_PHASH_THRESHOLD:
                match_type = "STRONG_NEAR_DUPLICATE"

            elif phash_distance <= REVIEW_PHASH_THRESHOLD:
                match_type = "REVIEW_NEAR_DUPLICATE"

            else:
                continue

            mae = resized_mae(
                a["path"],
                b["path"],
            )

            results.append({
                "match_type": match_type,
                "split_a": a["split"],
                "class_a": a["class"],
                "path_a": str(a["path"]),
                "split_b": b["split"],
                "class_b": b["class"],
                "path_b": str(b["path"]),
                "phash_distance": phash_distance,
                "resized_mae": mae,
            })

    priority = {
        "EXACT_FILE": 0,
        "EXACT_PIXELS": 1,
        "STRONG_NEAR_DUPLICATE": 2,
        "REVIEW_NEAR_DUPLICATE": 3,
    }

    results.sort(
        key=lambda row: (
            priority[row["match_type"]],
            row["phash_distance"],
            row["resized_mae"],
        )
    )

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "match_type",
        "split_a",
        "class_a",
        "path_a",
        "split_b",
        "class_b",
        "path_b",
        "phash_distance",
        "resized_mae",
        "review_image",
    ]

    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for number, row in enumerate(
            results,
            start=1,
        ):
            review_image = ""

            if create_review_images:
                review_image = str(
                    create_review_pair(
                        number,
                        row,
                    )
                )

            output_row = dict(row)
            output_row[
                "review_image"
            ] = review_image

            writer.writerow(output_row)

    counts = defaultdict(int)

    for row in results:
        counts[row["match_type"]] += 1

    print()
    print(f"Cross-split pairs checked       : {checked_pairs}")
    print(f"Exact file duplicates           : {counts['EXACT_FILE']}")
    print(f"Exact decoded-pixel duplicates  : {counts['EXACT_PIXELS']}")
    print(
        "Strong near-duplicate candidates: "
        f"{counts['STRONG_NEAR_DUPLICATE']}"
    )
    print(
        "Review near-duplicate candidates: "
        f"{counts['REVIEW_NEAR_DUPLICATE']}"
    )

    return results


# ============================================================
# STAGE 3 — CLEAN raw_by_class
# ============================================================

def allowed_match_types() -> set[str]:
    result = {
        "EXACT_FILE",
        "EXACT_PIXELS",
        "STRONG_NEAR_DUPLICATE",
    }

    if AUTO_CLEAN_REVIEW_NEAR_DUPLICATES:
        result.add(
            "REVIEW_NEAR_DUPLICATE"
        )

    return result


def choose_path_to_remove(
    row: dict,
) -> tuple[str, str, Path] | None:

    split_a = row["split_a"]
    split_b = row["split_b"]

    priority_a = SPLIT_PRIORITY[split_a]
    priority_b = SPLIT_PRIORITY[split_b]

    if priority_a == priority_b:
        return None

    if priority_a > priority_b:
        return (
            split_a,
            row["class_a"],
            Path(row["path_a"]).resolve(),
        )

    return (
        split_b,
        row["class_b"],
        Path(row["path_b"]).resolve(),
    )


def build_removal_plan(
    results: list[dict],
) -> dict[Path, dict]:

    allowed = allowed_match_types()
    plan: dict[Path, dict] = {}

    for row in results:
        if row["match_type"] not in allowed:
            continue

        chosen = choose_path_to_remove(row)

        if chosen is None:
            continue

        split, class_name, path = chosen

        if split == "train":
            continue

        entry = plan.setdefault(
            path,
            {
                "split": split,
                "class": class_name,
                "rows": [],
            },
        )
        entry["rows"].append(row)

    return plan


def safe_destination(
    split: str,
    class_name: str,
    source: Path,
) -> Path:

    destination = (
        QUARANTINE_DIR
        / split
        / class_name
        / source.name
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not destination.exists():
        return destination

    counter = 1

    while True:
        candidate = destination.with_name(
            f"{destination.stem}_duplicate_{counter}"
            f"{destination.suffix}"
        )

        if not candidate.exists():
            return candidate

        counter += 1


def clean_duplicates(
    plan: dict[Path, dict],
) -> None:

    print()
    print("=" * 78)
    print("STAGE 3 — CLEAN DUPLICATES FROM raw_by_class")
    print("=" * 78)

    print("Priority rule: train > valid > test")
    print(
        "Automatic cleaning types: "
        + ", ".join(
            sorted(allowed_match_types())
        )
    )

    print(
        f"Unique lower-priority images to move: "
        f"{len(plan)}"
    )

    QUARANTINE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    removal_rows = []

    counts_by_split_class = {
        split: {
            class_name: 0
            for class_name in CLASSES
        }
        for split in ["valid", "test"]
    }

    for index, (source, info) in enumerate(
        sorted(
            plan.items(),
            key=lambda item: (
                SPLIT_PRIORITY[item[1]["split"]],
                item[1]["class"],
                str(item[0]),
            ),
        ),
        start=1,
    ):
        split = info["split"]
        class_name = info["class"]

        if not source.exists():
            print(
                f"[{index}/{len(plan)}] "
                f"[SKIP missing] {source}"
            )
            continue

        destination = safe_destination(
            split,
            class_name,
            source,
        )

        match_types = sorted({
            row["match_type"]
            for row in info["rows"]
        })

        matched_against = sorted({
            (
                row["split_b"]
                if row["split_a"] == split
                else row["split_a"]
            )
            for row in info["rows"]
        })

        shutil.move(
            str(source),
            str(destination),
        )

        counts_by_split_class[
            split
        ][
            class_name
        ] += 1

        print(
            f"[{index}/{len(plan)}] MOVED "
            f"{split}/{class_name}/{source.name}"
        )
        print(
            f"    -> {destination}"
        )

        removal_rows.append({
            "removed_split": split,
            "class": class_name,
            "source": str(source),
            "quarantine_destination": str(destination),
            "match_types": "; ".join(match_types),
            "matched_against": "; ".join(matched_against),
        })

    with REMOVAL_LOG_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        fieldnames = [
            "removed_split",
            "class",
            "source",
            "quarantine_destination",
            "match_types",
            "matched_against",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(removal_rows)

    print()
    print("Removed/moved by class:")

    for split in ["valid", "test"]:
        print(
            f"  {split}: "
            + " | ".join(
                f"{class_name}="
                f"{counts_by_split_class[split][class_name]}"
                for class_name in CLASSES
            )
        )


# ============================================================
# STAGE 4 — FINAL COUNTS
# ============================================================

def count_final_dataset() -> dict[str, dict[str, int]]:
    counts = {
        split: {
            class_name: 0
            for class_name in CLASSES
        }
        for split in SPLITS
    }

    for split in SPLITS:
        for class_name in CLASSES:
            class_dir = (
                CLASS_ROOT
                / split
                / class_name
            )

            counts[
                split
            ][
                class_name
            ] = sum(
                1
                for path in class_dir.rglob("*")
                if (
                    path.is_file()
                    and path.suffix.lower()
                    in SUPPORTED_EXTENSIONS
                )
            )

    return counts


def write_final_counts(
    counts: dict[str, dict[str, int]],
) -> None:

    with CLASS_DISTRIBUTION_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "split",
            "Unripe",
            "Ripe",
            "Overripe",
            "Rotten",
            "total",
        ])

        for split in SPLITS:
            values = [
                counts[split][class_name]
                for class_name in CLASSES
            ]

            writer.writerow([
                split,
                *values,
                sum(values),
            ])


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 78)
    print("FINAL BANANA RIPENESS DATASET PREPARATION")
    print("=" * 78)
    print(f"Original raw source : {RAW_ROOT}")
    print(f"Final working output: {CLASS_ROOT}")
    print()
    print(
        "Review-level near duplicates: "
        + (
            "AUTO-CLEAN ENABLED"
            if AUTO_CLEAN_REVIEW_NEAR_DUPLICATES
            else "REPORT ONLY"
        )
    )

    validate_raw_structure()
    prepare_output()

    organise_by_class()

    print()
    print("=" * 78)
    print("STAGE 2 — AUDIT raw_by_class BEFORE CLEANING")
    print("=" * 78)

    before_results = audit_classified_dataset(
        output_csv=BEFORE_CSV,
        create_review_images=True,
    )

    plan = build_removal_plan(
        before_results
    )

    clean_duplicates(plan)

    print()
    print("=" * 78)
    print("STAGE 4 — VERIFY raw_by_class AFTER CLEANING")
    print("=" * 78)

    after_results = audit_classified_dataset(
        output_csv=AFTER_CSV,
        create_review_images=False,
    )

    counts = count_final_dataset()
    write_final_counts(counts)

    print()
    print("=" * 78)
    print("FINAL CLASS DISTRIBUTION")
    print("=" * 78)

    grand_total = 0

    for split in SPLITS:
        values = [
            counts[split][class_name]
            for class_name in CLASSES
        ]
        total = sum(values)
        grand_total += total

        print(
            f"{split:5s}: "
            f"Unripe={values[0]:4d} | "
            f"Ripe={values[1]:4d} | "
            f"Overripe={values[2]:4d} | "
            f"Rotten={values[3]:4d} | "
            f"TOTAL={total:4d}"
        )

    print()
    print(f"Total final clean images : {grand_total}")
    print(f"Before-clean audit CSV   : {BEFORE_CSV}")
    print(f"After-clean audit CSV    : {AFTER_CSV}")
    print(f"Review-pair images       : {REVIEW_DIR}")
    print(f"Quarantine folder        : {QUARANTINE_DIR}")
    print(f"Removal log              : {REMOVAL_LOG_CSV}")
    print(f"Final class-count CSV    : {CLASS_DISTRIBUTION_CSV}")

    print()
    if after_results:
        print(
            "[WARNING] Cross-split duplicate candidates still remain "
            "after cleaning."
        )
        print(
            "Inspect cross_split_duplicates_after_cleaning.csv "
            "before continuing."
        )
    else:
        print(
            "[OK] No configured cross-split duplicate candidates "
            "remain after cleaning."
        )

    print()
    print("NEXT:")
    print(
        "  Generate data/segmented FROM the final "
        "data/raw_by_class dataset."
    )
    print(
        "  Baseline evaluation should use data/raw_by_class."
    )
    print(
        "  Segmented evaluation should use data/segmented."
    )


if __name__ == "__main__":
    main()
