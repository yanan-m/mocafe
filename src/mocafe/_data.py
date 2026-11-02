"""CSV manifests of the prepared crops."""

from __future__ import annotations

import csv
from pathlib import Path

TARGET_ADAPT = "target_adapt.csv"  # unlabelled target pool: image, track_id
TARGET_TEST = "target_test.csv"  # labelled target test set: image, class_id
PUBLIC = "public_calibration.csv"  # labelled public set: image, track_id, class_id


def read_manifest(path: Path, columns: tuple[str, ...]) -> list[dict]:
    """Rows of a CSV manifest, checked to be non-empty and to contain ``columns``."""
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = set(columns) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} lacks column(s): {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def class_ids(rows: list[dict], num_classes: int, path: Path) -> list[int]:
    """Integer ``class_id`` of every row, checked against the model's number of classes."""
    labels = [int(r["class_id"]) for r in rows]
    if min(labels) < 0 or max(labels) >= num_classes:
        raise ValueError(f"{path}: class_id must lie in [0, {num_classes})")
    return labels
