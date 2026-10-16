"""Turn labelled snapshots into an Ultralytics dataset for training/finetune.py.

Input: a folder, such as the security camera viewer's training export
(Settings -> Object detection), unzipped:

    classes.txt              one class name per line (line N = class N)
    images/*.jpg             the snapshots
    labels/*.txt             YOLO labels, same file name; empty = nothing in it

Output: <out>/images/{train,val}, <out>/labels/{train,val}, <out>/dataset.yaml

Example:
    python -m training.prepare --source export/ --out datasets/home \\
        --auto-label yolo11n.pt --keep person,cat,dog

- The split is by day (the YYYYMMDD in the file name), never frame by
  frame: frames of one day look alike, so a random split leaks them into
  validation and overstates the score.
- --auto-label WEIGHTS --keep CLASSES labels those classes with an existing
  model wherever your labels don't already have them (pseudo-labels). Then
  fine-tuning on a new class (your cat "luna", say) doesn't make the model
  forget the classes it knew.
- It prints per-class counts and warns about what usually hurts: too few
  examples of a class, or no empty frames (the model then sees things
  everywhere).
"""

from __future__ import annotations

import argparse
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
MIN_PER_CLASS = 50  # fewer labelled examples than this rarely trains well
MIN_BACKGROUND = 0.05  # share of empty frames below which false positives grow
DAY = re.compile(r"(\d{8})T\d{6}Z")

Box = tuple[int, float, float, float, float]  # class, x_center, y_center, width, height (0-1)


def day_of(stem: str) -> str:
    match = DAY.search(stem)
    return match.group(1) if match else "unknown"


def split_by_day(stems: list[str], val_fraction: float = 0.2) -> tuple[list[str], list[str]]:
    """(train, val): whole days, the newest ones for validation (closest to
    what the camera will see next). With a single day, its last frames."""
    by_day: dict[str, list[str]] = defaultdict(list)
    for stem in sorted(stems):
        by_day[day_of(stem)].append(stem)
    days = sorted(by_day)
    if len(days) < 2:
        ordered = sorted(stems)
        if len(ordered) < 2:
            return ordered, []
        cut = len(ordered) - max(1, round(len(ordered) * val_fraction))
        return ordered[:cut], ordered[cut:]
    val: list[str] = []
    while days[1:] and len(val) < val_fraction * len(stems):
        val += by_day[days.pop()]
    train = [s for d in days for s in by_day[d]]
    return train, sorted(val)


def read_labels(path: Path) -> list[Box]:
    boxes: list[Box] = []
    if path.exists():
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) == 5:
                boxes.append((int(parts[0]), *(float(v) for v in parts[1:])))
    return boxes


def write_labels(path: Path, boxes: list[Box]) -> None:
    path.write_text("".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in boxes))


def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a[1] - a[3] / 2, a[2] - a[4] / 2, a[1] + a[3] / 2, a[2] + a[4] / 2
    bx1, by1, bx2, by2 = b[1] - b[3] / 2, b[2] - b[4] / 2, b[1] + b[3] / 2, b[2] + b[4] / 2
    w, h = min(ax2, bx2) - max(ax1, bx1), min(ay2, by2) - max(ay1, by1)
    if w <= 0 or h <= 0:
        return 0.0
    shared = w * h
    return shared / (a[3] * a[4] + b[3] * b[4] - shared)


def merge_pseudo(human: list[Box], model: list[Box], same: float = 0.5) -> list[Box]:
    """Your boxes, plus the model's where you have none of that class there."""
    added = [m for m in model if not any(h[0] == m[0] and iou(h, m) >= same for h in human)]
    return human + added


def model_boxes(model, image: Path, class_ids: dict[str, int], conf: float) -> list[Box]:
    """The model's boxes for the kept classes, as dataset class ids."""
    boxes: list[Box] = []
    for result in model.predict(source=str(image), conf=conf, verbose=False):
        for cls, xywhn in zip(result.boxes.cls, result.boxes.xywhn, strict=True):
            name = model.names[int(cls)]
            if name in class_ids:
                boxes.append((class_ids[name], *(float(v) for v in xywhn)))
    return boxes


def prepare(
    source: Path,
    out: Path,
    val_fraction: float = 0.2,
    model=None,
    keep: list[str] | None = None,
    conf: float = 0.4,
) -> dict:
    names = [n.strip() for n in (source / "classes.txt").read_text().splitlines() if n.strip()]
    for name in keep or []:  # classes kept with pseudo-labels join the list
        if name not in names:
            names.append(name)
    class_ids = {name: i for i, name in enumerate(names)}
    images = {
        p.stem: p
        for p in sorted((source / "images").iterdir())
        if p.suffix.lower() in IMAGE_SUFFIXES
    }
    train, val = split_by_day(list(images), val_fraction)

    counts: dict[str, Counter] = {"train": Counter(), "val": Counter()}
    empty = Counter()
    for split, stems in (("train", train), ("val", val)):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for stem in stems:
            image = images[stem]
            boxes = read_labels(source / "labels" / f"{stem}.txt")
            if model is not None and keep:
                kept = {class_ids[k] for k in keep}
                found = model_boxes(model, image, class_ids, conf)
                boxes = merge_pseudo(boxes, [b for b in found if b[0] in kept])
            shutil.copy2(image, out / "images" / split / image.name)
            write_labels(out / "labels" / split / f"{stem}.txt", boxes)
            counts[split].update(names[b[0]] for b in boxes)
            empty[split] += not boxes

    (out / "dataset.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n\nnames:\n"
        + "".join(f"  {i}: {name}\n" for i, name in enumerate(names))
    )
    warnings = []
    for name in names:
        if counts["train"][name] < MIN_PER_CLASS:
            warnings.append(
                f"only {counts['train'][name]} '{name}' boxes in train: aim for {MIN_PER_CLASS}+ "
                "(different days, light, distances)"
            )
        if counts["val"][name] == 0:
            warnings.append(f"no '{name}' in validation: its score can't be measured")
    if train and empty["train"] / len(train) < MIN_BACKGROUND:
        warnings.append(
            f"only {empty['train']} empty frames in train: add some (about 10%) "
            "so the model learns what is not a person or a pet"
        )
    if len({day_of(s) for s in images}) < 2:
        warnings.append("all images are from one day: validation shares its light and scene")
    return {
        "names": names,
        "train": len(train),
        "val": len(val),
        "counts": counts,
        "empty": empty,
        "warnings": warnings,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--source", required=True, type=Path, help="classes.txt + images/ + labels/")
    p.add_argument("--out", required=True, type=Path, help="Dataset folder to write")
    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Share of images (whole days) for validation",
    )
    p.add_argument(
        "--auto-label", metavar="WEIGHTS", help="Model that pseudo-labels the --keep classes"
    )
    p.add_argument("--keep", default="", help="Classes to keep, e.g. person,cat,dog")
    p.add_argument("--conf", type=float, default=0.4, help="Minimum score of a pseudo-label")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    keep = [k.strip() for k in args.keep.split(",") if k.strip()]
    model = None
    if args.auto_label:
        from ultralytics import YOLO

        model = YOLO(args.auto_label)
    report = prepare(args.source, args.out, args.val_fraction, model, keep, args.conf)
    print(f"Classes: {', '.join(report['names'])}")
    print(f"Images: {report['train']} train, {report['val']} val (split by day)")
    for split in ("train", "val"):
        boxes = ", ".join(f"{n} {report['counts'][split][n]}" for n in report["names"])
        print(f"  {split}: {boxes}; empty frames {report['empty'][split]}")
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")
    data = args.out / "dataset.yaml"
    print(f"\nNext: python -m training.finetune --data {data} --preset indoor-camera")
    return report


if __name__ == "__main__":
    main()
