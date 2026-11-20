"""Datasets for annotation and fine-tuning, stored as plain folders.

    <data_dir>/datasets/<id>/meta.json
    <data_dir>/datasets/<id>/images/<file>
    <data_dir>/datasets/<id>/labels/<stem>.txt   (YOLO format, normalised 0-1)

A missing label file means "not annotated yet"; an empty one means
"annotated, nothing in it" (a useful background image).
"""

from __future__ import annotations

import io
import json
import math
import re
import shutil
import time
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image

from app.imaging import decode_image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


class DatasetError(ValueError):
    pass


class NotFound(LookupError):
    pass


def safe_name(name: str) -> str:
    cleaned = _SAFE.sub("_", Path(name).name).strip("._")
    if not cleaned:
        raise DatasetError(f"invalid name {name!r}")
    return cleaned


class DatasetStore:
    def __init__(self, root: Path, max_pixels: int = 40_000_000) -> None:
        self.root = root
        self.max_pixels = max_pixels

    # --- datasets ---------------------------------------------------------
    def path(self, dataset_id: str) -> Path:
        path = self.root / safe_name(dataset_id)
        if not (path / "meta.json").is_file():
            raise NotFound(f"dataset {dataset_id!r} not found")
        return path

    def meta(self, dataset_id: str) -> dict:
        return json.loads((self.path(dataset_id) / "meta.json").read_text())

    def list(self) -> list[dict]:
        if not self.root.exists():
            return []
        out = []
        for d in sorted(self.root.iterdir()):
            if (d / "meta.json").is_file():
                meta = json.loads((d / "meta.json").read_text())
                images = self._images(d)
                meta["images"] = len(images)
                meta["annotated"] = sum((d / "labels" / f"{p.stem}.txt").exists() for p in images)
                out.append(meta)
        return out

    def create(
        self, name: str, classes: list[str], task: str = "detect", description: str = ""
    ) -> dict:
        if task not in {"detect", "pose"}:
            raise DatasetError("task must be 'detect' or 'pose'")
        classes = [c.strip() for c in classes if c.strip()]
        if not classes:
            raise DatasetError("a dataset needs at least one class")
        if len(set(classes)) != len(classes):
            raise DatasetError("class names must be unique")
        dataset_id = safe_name(name).lower()
        path = self.root / dataset_id
        if path.exists():
            raise DatasetError(f"dataset {dataset_id!r} already exists")
        (path / "images").mkdir(parents=True)
        (path / "labels").mkdir()
        meta = {
            "id": dataset_id,
            "name": name,
            "task": task,
            "classes": classes,
            "description": description,
            "created": time.time(),
        }
        (path / "meta.json").write_text(json.dumps(meta, indent=2))
        return meta

    def update_classes(self, dataset_id: str, classes: list[str]) -> dict:
        """Rename or append classes. Removing classes would renumber labels: refused."""
        meta = self.meta(dataset_id)
        classes = [c.strip() for c in classes if c.strip()]
        if len(classes) < len(meta["classes"]):
            raise DatasetError("classes can be renamed or added, not removed")
        meta["classes"] = classes
        (self.path(dataset_id) / "meta.json").write_text(json.dumps(meta, indent=2))
        return meta

    def delete(self, dataset_id: str) -> None:
        shutil.rmtree(self.path(dataset_id))

    # --- images -----------------------------------------------------------
    @staticmethod
    def _images(path: Path) -> list[Path]:
        return sorted(p for p in (path / "images").iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)

    def images(self, dataset_id: str) -> list[dict]:
        path = self.path(dataset_id)
        out = []
        for p in self._images(path):
            label = path / "labels" / f"{p.stem}.txt"
            boxes = self._read(label) if label.exists() else None
            out.append(
                {
                    "file": p.name,
                    "annotated": boxes is not None,
                    "objects": len(boxes) if boxes is not None else 0,
                }
            )
        return out

    def add_image(self, dataset_id: str, filename: str, data: bytes) -> str:
        path = self.path(dataset_id)
        image = decode_image(data, max_pixels=self.max_pixels)  # validates
        stem = Path(safe_name(filename)).stem
        target = path / "images" / f"{stem}.jpg"
        n = 1
        while target.exists() or (path / "labels" / f"{target.stem}.txt").exists():
            target = path / "images" / f"{stem}_{n}.jpg"
            n += 1
        image.save(target, format="JPEG", quality=95)
        return target.name

    def image_file(self, dataset_id: str, filename: str) -> Path:
        p = self.path(dataset_id) / "images" / safe_name(filename)
        if not p.is_file():
            raise NotFound(f"image {filename!r} not found")
        return p

    def delete_image(self, dataset_id: str, filename: str) -> None:
        p = self.image_file(dataset_id, filename)
        (p.parent.parent / "labels" / f"{p.stem}.txt").unlink(missing_ok=True)
        p.unlink()

    # --- labels -----------------------------------------------------------
    @staticmethod
    def _read(label: Path) -> list[dict]:
        boxes = []
        for line in label.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            box = {
                "cls": int(parts[0]),
                "x": float(parts[1]),
                "y": float(parts[2]),
                "w": float(parts[3]),
                "h": float(parts[4]),
            }
            rest = [float(v) for v in parts[5:]]
            if rest and len(rest) % 3 == 0:
                box["keypoints"] = [rest[i : i + 3] for i in range(0, len(rest), 3)]
            boxes.append(box)
        return boxes

    def labels(self, dataset_id: str, filename: str) -> list[dict] | None:
        image = self.image_file(dataset_id, filename)
        label = image.parent.parent / "labels" / f"{image.stem}.txt"
        return self._read(label) if label.exists() else None

    def save_labels(self, dataset_id: str, filename: str, boxes: list[dict]) -> int:
        meta = self.meta(dataset_id)
        image = self.image_file(dataset_id, filename)
        n_classes = len(meta["classes"])
        lines = []
        for b in boxes:
            cls = int(b["cls"])
            if not 0 <= cls < n_classes:
                raise DatasetError(f"class id {cls} out of range (0-{n_classes - 1})")
            x, y, w, h = (min(1.0, max(0.0, float(b[k]))) for k in ("x", "y", "w", "h"))
            if w <= 0 or h <= 0:
                continue
            fields = [str(cls), *(f"{v:.6f}" for v in (x, y, w, h))]
            if meta["task"] == "pose":
                kpts = b.get("keypoints") or [[0.0, 0.0, 0.0]] * 17
                if len(kpts) != 17:
                    raise DatasetError("pose datasets need 17 COCO keypoints per object")
                fields += [f"{float(v):.6f}" for kp in kpts for v in kp]
            lines.append(" ".join(fields))
        label = image.parent.parent / "labels" / f"{image.stem}.txt"
        label.write_text("".join(f"{line}\n" for line in lines))
        return len(lines)

    def clear_labels(self, dataset_id: str, filename: str) -> None:
        image = self.image_file(dataset_id, filename)
        (image.parent.parent / "labels" / f"{image.stem}.txt").unlink(missing_ok=True)

    # --- statistics ---------------------------------------------------------
    def stats(self, dataset_id: str, weight_power: float = 0.5) -> dict:
        """Class balance of the annotated images, with suggested class weights.

        Suggested weight of class c: ``(N / (K * n_c)) ** power`` normalised to a
        mean of 1 (power 1 = inverse frequency, 0.5 = its softer square root).
        """
        meta = self.meta(dataset_id)
        path = self.path(dataset_id)
        names = meta["classes"]
        boxes = Counter()
        images_with = Counter()
        annotated = background = 0
        sizes = []
        all_images = self._images(path)
        for p in all_images:
            label = path / "labels" / f"{p.stem}.txt"
            if not label.exists():
                continue
            annotated += 1
            objects = self._read(label)
            if not objects:
                background += 1
            for b in objects:
                boxes[b["cls"]] += 1
                sizes.append(b["w"] * b["h"])
            images_with.update({b["cls"] for b in objects})

        counts = {name: boxes.get(i, 0) for i, name in enumerate(names)}
        present = [c for c in counts.values() if c > 0]
        total = sum(present)
        raw = [(total / (len(names) * c)) ** weight_power if c else 0.0 for c in counts.values()]
        nonzero = [w for w in raw if w > 0]
        mean = sum(nonzero) / len(nonzero) if nonzero else 1.0
        weights = [round(w / mean, 3) if w else 1.0 for w in raw]

        warnings = []
        for name, c in counts.items():
            if c < 50:
                warnings.append(f"'{name}' has {c} boxes: aim for 50+ (ideally 200+) per class")
        if present and max(present) / min(present) > 5:
            warnings.append(
                "strong class imbalance (ratio "
                f"{max(present) / min(present):.1f}): consider repeat-factor sampling, "
                "class weights or a focal classification loss"
            )
        if annotated and background / annotated < 0.05:
            warnings.append("fewer than 5% background images: add a few empty frames")
        small = sum(s < 0.01 for s in sizes)
        return {
            "classes": names,
            "images": len(all_images),
            "annotated": annotated,
            "unannotated": len(all_images) - annotated,
            "background": background,
            "boxes": counts,
            "images_with_class": {name: images_with.get(i, 0) for i, name in enumerate(names)},
            "imbalance_ratio": round(max(present) / min(present), 2) if present else None,
            "suggested_class_weights": weights,
            "small_objects_share": round(small / len(sizes), 3) if sizes else 0.0,
            "warnings": warnings,
        }

    # --- import / export ------------------------------------------------------
    def import_zip(self, dataset_id: str, data: bytes) -> dict:
        """Import a YOLO-format zip: images/... and labels/... (any depth),
        optionally classes.txt. Labels must use this dataset's class ids."""
        path = self.path(dataset_id)
        added = labelled = 0
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            members = {Path(n): n for n in zf.namelist() if not n.endswith("/")}
            labels = {
                p.stem: n for p, n in members.items() if p.suffix == ".txt" and "labels" in p.parts
            }
            for p, n in members.items():
                if p.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                saved = self.add_image(dataset_id, p.name, zf.read(n))
                added += 1
                if p.stem in labels:
                    (path / "labels" / f"{Path(saved).stem}.txt").write_bytes(
                        zf.read(labels[p.stem])
                    )
                    labelled += 1
        return {"images": added, "labelled": labelled}

    def export_zip(self, dataset_id: str) -> bytes:
        path = self.path(dataset_id)
        meta = self.meta(dataset_id)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("classes.txt", "\n".join(meta["classes"]) + "\n")
            for p in self._images(path):
                zf.write(p, f"images/{p.name}")
                label = path / "labels" / f"{p.stem}.txt"
                if label.exists():
                    zf.write(label, f"labels/{label.name}")
        return buf.getvalue()


def thumbnail(path: Path, size: int = 256) -> bytes:
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((size, size))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=80)
        return buf.getvalue()


def to_normalised(box: tuple[float, float, float, float], width: int, height: int) -> dict:
    x1, y1, x2, y2 = box
    return {
        "x": (x1 + x2) / 2 / width,
        "y": (y1 + y2) / 2 / height,
        "w": (x2 - x1) / width,
        "h": (y2 - y1) / height,
    }


def nan_to_none(value):
    return None if isinstance(value, float) and math.isnan(value) else value
