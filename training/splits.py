"""Build a train/val split from a lab dataset, with optional rebalancing.

A lab dataset (``app/lab/datasets.py``) is a folder:

    meta.json            {"name", "task", "classes", ...}
    images/<file>        the pictures
    labels/<stem>.txt    YOLO labels; a missing file means "not annotated yet",
                         an empty file means "annotated, nothing in it"

:func:`build_split` writes ``train.txt`` / ``val.txt`` (absolute image paths,
one per line; Ultralytics finds the labels by replacing ``/images/`` with
``/labels/``) and a ``dataset.yaml`` next to them. No file is copied.

Class imbalance: with ``rebalance="repeat-factor"`` images that contain rare
classes are listed several times in ``train.txt`` (repeat-factor sampling,
Gupta et al., LVIS 2019): class ``c`` gets ``r_c = max(1, sqrt(t / f_c))``
where ``f_c`` is the share of training images containing ``c``; an image is
repeated ``max r_c`` over its classes (capped by ``max_repeat``).
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path

from training.prepare import IMAGE_SUFFIXES, read_labels, split_by_day

# COCO-17 keypoints, the layout of the yolo11*-pose models.
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]  # fmt: skip
# Index of each keypoint's mirror image, used by the left-right flip augmentation.
COCO_FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def labelled_images(dataset: Path) -> list[Path]:
    images = sorted(p for p in (dataset / "images").iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    return [p for p in images if (dataset / "labels" / f"{p.stem}.txt").exists()]


def image_classes(dataset: Path, image: Path) -> set[int]:
    path = dataset / "labels" / f"{image.stem}.txt"
    classes = set()
    for line in path.read_text().splitlines():
        if line.split():
            classes.add(int(line.split()[0]))
    return classes


def repeat_factors(
    per_image: list[set[int]], threshold: float = 0.3, max_repeat: int = 4
) -> list[int]:
    """Integer repeat count per image (repeat-factor sampling)."""
    n = len(per_image) or 1
    freq = Counter(c for classes in per_image for c in classes)
    r_class = {c: max(1.0, math.sqrt(threshold / (k / n))) for c, k in freq.items()}
    return [
        min(max_repeat, round(max((r_class[c] for c in classes), default=1.0)))
        for classes in per_image
    ]


def build_split(
    dataset: Path,
    out: Path,
    *,
    val_fraction: float = 0.2,
    strategy: str = "random",
    seed: int = 0,
    rebalance: str = "none",
    rfs_threshold: float = 0.3,
    max_repeat: int = 4,
) -> dict:
    meta = json.loads((dataset / "meta.json").read_text())
    names: list[str] = meta["classes"]
    images = labelled_images(dataset)
    if len(images) < 2:
        raise ValueError(f"need at least 2 annotated images, the dataset has {len(images)}")

    if strategy == "by-day":
        by_stem = {p.stem: p for p in images}
        train_stems, val_stems = split_by_day(list(by_stem), val_fraction)
        train = [by_stem[s] for s in train_stems]
        val = [by_stem[s] for s in val_stems]
    else:
        shuffled = images[:]
        random.Random(seed).shuffle(shuffled)
        n_val = max(1, round(len(shuffled) * val_fraction))
        val, train = sorted(shuffled[:n_val]), sorted(shuffled[n_val:])

    per_image = [image_classes(dataset, p) for p in train]
    repeats = (
        repeat_factors(per_image, rfs_threshold, max_repeat)
        if rebalance == "repeat-factor"
        else [1] * len(train)
    )
    train_list = [p for p, r in zip(train, repeats, strict=True) for _ in range(r)]

    out.mkdir(parents=True, exist_ok=True)
    (out / "train.txt").write_text("".join(f"{p.resolve()}\n" for p in train_list))
    (out / "val.txt").write_text("".join(f"{p.resolve()}\n" for p in val))
    yaml = f"path: {out.resolve()}\ntrain: train.txt\nval: val.txt\n"
    if meta.get("task") == "pose":
        yaml += f"kpt_shape: [{len(COCO_KEYPOINTS)}, 3]\nflip_idx: {COCO_FLIP_IDX}\n"
    yaml += "\nnames:\n" + "".join(f"  {i}: {json.dumps(n)}\n" for i, n in enumerate(names))
    (out / "dataset.yaml").write_text(yaml)

    def counts(paths: list[Path]) -> dict[str, int]:
        c = Counter()
        for p in paths:
            for box in read_labels_any(dataset / "labels" / f"{p.stem}.txt"):
                c[names[box] if box < len(names) else str(box)] += 1
        return {n: c.get(n, 0) for n in names}

    return {
        "yaml": str(out / "dataset.yaml"),
        "train_images": len(train),
        "train_entries": len(train_list),
        "val_images": len(val),
        "train_counts": counts(train_list),
        "val_counts": counts(val),
    }


def read_labels_any(path: Path) -> list[int]:
    """Class ids of every object in a label file (detect or pose format)."""
    if not path.exists():
        return []
    return [int(line.split()[0]) for line in path.read_text().splitlines() if line.split()]


__all__ = [
    "COCO_FLIP_IDX",
    "COCO_KEYPOINTS",
    "build_split",
    "labelled_images",
    "read_labels",
    "repeat_factors",
]
