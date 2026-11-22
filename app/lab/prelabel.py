"""Pre-label dataset images with the active model: correcting boxes is much
faster than drawing them."""

from __future__ import annotations

from PIL import Image

from app.detectors import Detection
from app.lab.datasets import DatasetStore, to_normalised
from app.service import DetectionService

# YOLO pose label visibility: 0 = not labelled, 1 = labelled but hidden, 2 = visible.
VISIBLE_THRESHOLD = 0.5


def classes_known_by_model(dataset_classes: list[str], service: DetectionService) -> list[str]:
    model_classes = {name.lower() for name in service.detector.labels.values()}
    return [name for name in dataset_classes if name.lower() in model_classes]


def prelabel_dataset(
    store: DatasetStore,
    service: DetectionService,
    dataset_id: str,
    *,
    min_confidence: float,
    overwrite: bool,
    only_files: list[str] | None = None,
) -> dict:
    """Label the dataset's images with the model's detections of the dataset classes.

    Images that already have labels are skipped unless ``overwrite`` is set.
    """
    dataset_classes = store.metadata(dataset_id)["classes"]
    class_index_by_name = {name.lower(): index for index, name in enumerate(dataset_classes)}
    usable_classes = classes_known_by_model(dataset_classes, service)
    labelled_count = 0
    for item in store.images(dataset_id):
        if only_files and item["file"] not in only_files:
            continue
        if item["annotated"] and not overwrite:
            continue
        with Image.open(store.image_file(dataset_id, item["file"])) as opened:
            image = opened.convert("RGB")
        result = service.detect_in_image(
            image, class_names=usable_classes, min_confidence=min_confidence
        )
        labels = [
            _to_label(detection, class_index_by_name, image.width, image.height)
            for detection in result.detections
        ]
        store.save_labels(dataset_id, item["file"], labels)
        labelled_count += 1
    return {"labelled": labelled_count, "classes": usable_classes}


def _to_label(
    detection: Detection, class_index_by_name: dict[str, int], width: int, height: int
) -> dict:
    label = {
        "cls": class_index_by_name[detection.label.lower()],
        **to_normalised(detection.box, width, height),
    }
    if detection.keypoints:
        label["keypoints"] = [
            [x / width, y / height, _yolo_visibility(visibility)]
            for x, y, visibility in detection.keypoints
        ]
    return label


def _yolo_visibility(score: float) -> float:
    if score > VISIBLE_THRESHOLD:
        return 2.0
    return 1.0 if score > 0 else 0.0
