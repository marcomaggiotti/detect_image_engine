"""Datasets for annotation and fine-tuning, stored as plain folders.

    <data_dir>/datasets/<id>/meta.json
    <data_dir>/datasets/<id>/images/<file>
    <data_dir>/datasets/<id>/labels/<stem>.txt   (YOLO format, normalised 0-1)

A missing label file means "not annotated yet"; an empty one means
"annotated, nothing in it" (a useful background image).

Label line (detection): ``class x_center y_center width height``; pose datasets add
17 COCO keypoints as ``x y visibility`` triples.
"""

from __future__ import annotations

import io
import json
import re
import shutil
import time
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image

from app.imaging import decode_image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
COCO_KEYPOINT_COUNT = 17
MIN_BOXES_PER_CLASS = 50
MAX_IMBALANCE_RATIO = 5
MIN_BACKGROUND_SHARE = 0.05
SMALL_OBJECT_AREA = 0.01  # share of the picture
_UNSAFE_CHARACTERS = re.compile(r"[^A-Za-z0-9._-]+")


class DatasetError(ValueError):
    """Invalid dataset operation (bad name, class id out of range, ...)."""


class DatasetNotFoundError(LookupError):
    """The dataset or one of its images does not exist."""


def safe_name(name: str) -> str:
    """A file-system safe version of ``name`` (no path, no special characters)."""
    cleaned = _UNSAFE_CHARACTERS.sub("_", Path(name).name).strip("._")
    if not cleaned:
        raise DatasetError(f"invalid name {name!r}")
    return cleaned


def _clean_class_names(class_names: list[str]) -> list[str]:
    return [name.strip() for name in class_names if name.strip()]


def _label_path(image_path: Path) -> Path:
    """``<dataset>/images/x.jpg`` -> ``<dataset>/labels/x.txt``."""
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


class DatasetStore:
    def __init__(self, root: Path, max_pixels: int = 40_000_000) -> None:
        self.root = root
        self.max_pixels = max_pixels

    # --- datasets -------------------------------------------------------------------
    def path(self, dataset_id: str) -> Path:
        dataset_path = self.root / safe_name(dataset_id)
        if not (dataset_path / "meta.json").is_file():
            raise DatasetNotFoundError(f"dataset {dataset_id!r} not found")
        return dataset_path

    def metadata(self, dataset_id: str) -> dict:
        return json.loads((self.path(dataset_id) / "meta.json").read_text())

    def _write_metadata(self, dataset_id: str, metadata: dict) -> None:
        (self.path(dataset_id) / "meta.json").write_text(json.dumps(metadata, indent=2))

    def list(self) -> list[dict]:
        """Every dataset with its number of images and of annotated images."""
        if not self.root.exists():
            return []
        datasets = []
        for dataset_path in sorted(self.root.iterdir()):
            if not (dataset_path / "meta.json").is_file():
                continue
            metadata = json.loads((dataset_path / "meta.json").read_text())
            image_paths = self._image_paths(dataset_path)
            metadata["images"] = len(image_paths)
            metadata["annotated"] = sum(_label_path(image).exists() for image in image_paths)
            datasets.append(metadata)
        return datasets

    def create(
        self, name: str, class_names: list[str], task: str = "detect", description: str = ""
    ) -> dict:
        if task not in {"detect", "pose"}:
            raise DatasetError("task must be 'detect' or 'pose'")
        class_names = _clean_class_names(class_names)
        if not class_names:
            raise DatasetError("a dataset needs at least one class")
        if len(set(class_names)) != len(class_names):
            raise DatasetError("class names must be unique")
        dataset_id = safe_name(name).lower()
        dataset_path = self.root / dataset_id
        if dataset_path.exists():
            raise DatasetError(f"dataset {dataset_id!r} already exists")
        (dataset_path / "images").mkdir(parents=True)
        (dataset_path / "labels").mkdir()
        metadata = {
            "id": dataset_id,
            "name": name,
            "task": task,
            "classes": class_names,
            "description": description,
            "created": time.time(),
        }
        (dataset_path / "meta.json").write_text(json.dumps(metadata, indent=2))
        return metadata

    def update_classes(self, dataset_id: str, class_names: list[str]) -> dict:
        """Rename or append classes. Removing classes would renumber labels: refused."""
        metadata = self.metadata(dataset_id)
        class_names = _clean_class_names(class_names)
        if len(class_names) < len(metadata["classes"]):
            raise DatasetError("classes can be renamed or added, not removed")
        metadata["classes"] = class_names
        self._write_metadata(dataset_id, metadata)
        return metadata

    def delete(self, dataset_id: str) -> None:
        shutil.rmtree(self.path(dataset_id))

    # --- images ---------------------------------------------------------------------
    @staticmethod
    def _image_paths(dataset_path: Path) -> list[Path]:
        return sorted(
            path
            for path in (dataset_path / "images").iterdir()
            if path.suffix.lower() in IMAGE_SUFFIXES
        )

    def images(self, dataset_id: str) -> list[dict]:
        """Every image with its annotation status and number of objects."""
        summaries = []
        for image_path in self._image_paths(self.path(dataset_id)):
            label_path = _label_path(image_path)
            objects = self._read_label_file(label_path) if label_path.exists() else None
            summaries.append(
                {
                    "file": image_path.name,
                    "annotated": objects is not None,
                    "objects": len(objects) if objects is not None else 0,
                }
            )
        return summaries

    def add_image(self, dataset_id: str, filename: str, content: bytes) -> str:
        """Validate and store an image as JPEG; returns its file name (made unique)."""
        dataset_path = self.path(dataset_id)
        image = decode_image(content, max_pixels=self.max_pixels)
        stem = Path(safe_name(filename)).stem
        target = dataset_path / "images" / f"{stem}.jpg"
        suffix_number = 1
        while target.exists() or _label_path(target).exists():
            target = dataset_path / "images" / f"{stem}_{suffix_number}.jpg"
            suffix_number += 1
        image.save(target, format="JPEG", quality=95)
        return target.name

    def image_file(self, dataset_id: str, filename: str) -> Path:
        image_path = self.path(dataset_id) / "images" / safe_name(filename)
        if not image_path.is_file():
            raise DatasetNotFoundError(f"image {filename!r} not found")
        return image_path

    def delete_image(self, dataset_id: str, filename: str) -> None:
        image_path = self.image_file(dataset_id, filename)
        _label_path(image_path).unlink(missing_ok=True)
        image_path.unlink()

    # --- labels ---------------------------------------------------------------------
    @staticmethod
    def _read_label_file(label_path: Path) -> list[dict]:
        objects = []
        for line in label_path.read_text().splitlines():
            values = line.split()
            if len(values) < 5:
                continue
            class_id, x_center, y_center, width, height = values[:5]
            label = {
                "cls": int(class_id),
                "x": float(x_center),
                "y": float(y_center),
                "w": float(width),
                "h": float(height),
            }
            keypoint_values = [float(value) for value in values[5:]]
            if keypoint_values and len(keypoint_values) % 3 == 0:
                label["keypoints"] = [
                    keypoint_values[start : start + 3]
                    for start in range(0, len(keypoint_values), 3)
                ]
            objects.append(label)
        return objects

    def labels(self, dataset_id: str, filename: str) -> list[dict] | None:
        """The image's objects, or ``None`` when it is not annotated yet."""
        label_path = _label_path(self.image_file(dataset_id, filename))
        return self._read_label_file(label_path) if label_path.exists() else None

    def save_labels(self, dataset_id: str, filename: str, objects: list[dict]) -> int:
        """Write the image's label file; returns the number of objects saved."""
        metadata = self.metadata(dataset_id)
        image_path = self.image_file(dataset_id, filename)
        class_count = len(metadata["classes"])
        lines = []
        for label in objects:
            class_id = int(label["cls"])
            if not 0 <= class_id < class_count:
                raise DatasetError(f"class id {class_id} out of range (0-{class_count - 1})")
            x_center, y_center, width, height = (
                min(1.0, max(0.0, float(label[key]))) for key in ("x", "y", "w", "h")
            )
            if width <= 0 or height <= 0:
                continue
            fields = [
                str(class_id),
                *(f"{value:.6f}" for value in (x_center, y_center, width, height)),
            ]
            if metadata["task"] == "pose":
                fields += self._keypoint_fields(label.get("keypoints"))
            lines.append(" ".join(fields))
        _label_path(image_path).write_text("".join(f"{line}\n" for line in lines))
        return len(lines)

    @staticmethod
    def _keypoint_fields(keypoints: list[list[float]] | None) -> list[str]:
        keypoints = keypoints or [[0.0, 0.0, 0.0]] * COCO_KEYPOINT_COUNT
        if len(keypoints) != COCO_KEYPOINT_COUNT:
            raise DatasetError(
                f"pose datasets need {COCO_KEYPOINT_COUNT} COCO keypoints per object"
            )
        return [f"{float(value):.6f}" for keypoint in keypoints for value in keypoint]

    def clear_labels(self, dataset_id: str, filename: str) -> None:
        _label_path(self.image_file(dataset_id, filename)).unlink(missing_ok=True)

    # --- statistics -----------------------------------------------------------------
    def stats(self, dataset_id: str, weight_power: float = 0.5) -> dict:
        """Class balance of the annotated images, with suggested class weights."""
        class_names = self.metadata(dataset_id)["classes"]
        image_paths = self._image_paths(self.path(dataset_id))
        boxes_per_class: Counter = Counter()
        images_per_class: Counter = Counter()
        object_areas: list[float] = []
        annotated_count = background_count = 0
        for image_path in image_paths:
            label_path = _label_path(image_path)
            if not label_path.exists():
                continue
            annotated_count += 1
            objects = self._read_label_file(label_path)
            if not objects:
                background_count += 1
            for label in objects:
                boxes_per_class[label["cls"]] += 1
                object_areas.append(label["w"] * label["h"])
            images_per_class.update({label["cls"] for label in objects})

        box_counts = {name: boxes_per_class.get(index, 0) for index, name in enumerate(class_names)}
        present_counts = [count for count in box_counts.values() if count > 0]
        imbalance_ratio = max(present_counts) / min(present_counts) if present_counts else None
        small_objects = sum(area < SMALL_OBJECT_AREA for area in object_areas)
        return {
            "classes": class_names,
            "images": len(image_paths),
            "annotated": annotated_count,
            "unannotated": len(image_paths) - annotated_count,
            "background": background_count,
            "boxes": box_counts,
            "images_with_class": {
                name: images_per_class.get(index, 0) for index, name in enumerate(class_names)
            },
            "imbalance_ratio": round(imbalance_ratio, 2) if imbalance_ratio else None,
            "suggested_class_weights": suggested_class_weights(
                list(box_counts.values()), weight_power
            ),
            "small_objects_share": round(small_objects / len(object_areas), 3)
            if object_areas
            else 0.0,
            "warnings": _balance_warnings(
                box_counts, imbalance_ratio, annotated_count, background_count
            ),
        }

    # --- import / export ------------------------------------------------------------
    def import_zip(self, dataset_id: str, archive: bytes) -> dict:
        """Import a YOLO-format zip: images/... and labels/... (any depth).

        Labels must use this dataset's class ids.
        """
        dataset_path = self.path(dataset_id)
        image_count = labelled_count = 0
        with zipfile.ZipFile(io.BytesIO(archive)) as zip_file:
            members = {Path(name): name for name in zip_file.namelist() if not name.endswith("/")}
            label_members = {
                path.stem: name
                for path, name in members.items()
                if path.suffix == ".txt" and "labels" in path.parts
            }
            for path, name in members.items():
                if path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                saved_name = self.add_image(dataset_id, path.name, zip_file.read(name))
                image_count += 1
                if path.stem in label_members:
                    saved_label = _label_path(dataset_path / "images" / saved_name)
                    saved_label.write_bytes(zip_file.read(label_members[path.stem]))
                    labelled_count += 1
        return {"images": image_count, "labelled": labelled_count}

    def export_zip(self, dataset_id: str) -> bytes:
        """The dataset as a YOLO-format zip (classes.txt, images/, labels/)."""
        class_names = self.metadata(dataset_id)["classes"]
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr("classes.txt", "\n".join(class_names) + "\n")
            for image_path in self._image_paths(self.path(dataset_id)):
                zip_file.write(image_path, f"images/{image_path.name}")
                label_path = _label_path(image_path)
                if label_path.exists():
                    zip_file.write(label_path, f"labels/{label_path.name}")
        return buffer.getvalue()


def suggested_class_weights(box_counts: list[int], power: float = 0.5) -> list[float]:
    """Class weight ``(N / (K * n_c)) ** power``, normalised to a mean of 1.

    ``power`` 1 is inverse frequency, 0.5 its softer square root. Classes without
    boxes get 1.
    """
    total = sum(box_counts)
    class_count = len(box_counts)
    raw_weights = [
        (total / (class_count * count)) ** power if count else 0.0 for count in box_counts
    ]
    nonzero = [weight for weight in raw_weights if weight > 0]
    mean_weight = sum(nonzero) / len(nonzero) if nonzero else 1.0
    return [round(weight / mean_weight, 3) if weight else 1.0 for weight in raw_weights]


def _balance_warnings(
    box_counts: dict[str, int],
    imbalance_ratio: float | None,
    annotated_count: int,
    background_count: int,
) -> list[str]:
    warnings = [
        f"'{name}' has {count} boxes: aim for {MIN_BOXES_PER_CLASS}+ (ideally 200+) per class"
        for name, count in box_counts.items()
        if count < MIN_BOXES_PER_CLASS
    ]
    if imbalance_ratio and imbalance_ratio > MAX_IMBALANCE_RATIO:
        warnings.append(
            f"strong class imbalance (ratio {imbalance_ratio:.1f}): consider repeat-factor "
            "sampling, class weights or a focal classification loss"
        )
    if annotated_count and background_count / annotated_count < MIN_BACKGROUND_SHARE:
        warnings.append("fewer than 5% background images: add a few empty frames")
    return warnings


def thumbnail(path: Path, max_side: int = 256) -> bytes:
    with Image.open(path) as opened:
        preview = opened.convert("RGB")
    preview.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    preview.save(buffer, format="JPEG", quality=80)
    return buffer.getvalue()


def to_normalised(box: tuple[float, float, float, float], width: int, height: int) -> dict:
    """Pixel corners -> YOLO centre/size normalised to the image size."""
    x1, y1, x2, y2 = box
    return {
        "x": (x1 + x2) / 2 / width,
        "y": (y1 + y2) / 2 / height,
        "w": (x2 - x1) / width,
        "h": (y2 - y1) / height,
    }
