"""Ultralytics backend: YOLOv8 / YOLO11 / RT-DETR and their fine-tuned weights.

Fine-tuned ``best.pt`` files carry their own class names, so pointing
``DETECT_MODEL_WEIGHTS`` at the output of ``training/finetune.py`` is enough.
Exported formats supported by Ultralytics (ONNX, TensorRT, OpenVINO...) load
the same way.
"""

from __future__ import annotations

import os
from collections.abc import Collection

from PIL import Image

from app.detectors.base import Detection, Detector, DetectorConfig, load_labels

_BOX_TASKS = {"detect", "segment", "pose"}


class UltralyticsDetector(Detector):
    backend_name = "ultralytics"

    def __init__(self, config: DetectorConfig) -> None:
        super().__init__(config)
        # Never let the library pip-install packages inside a running service.
        os.environ.setdefault("YOLO_AUTOINSTALL", "false")
        from ultralytics import RTDETR, YOLO

        architecture = (config.arch or "yolo").lower()
        if architecture == "rtdetr":
            self._model = RTDETR(config.weights)
        elif architecture == "yolo":
            self._model = YOLO(config.weights)
        else:
            raise ValueError(
                f"Unsupported ultralytics arch {config.arch!r} (use 'yolo' or 'rtdetr')"
            )

        if self._model.task not in _BOX_TASKS:
            raise ValueError(
                f"{config.weights!r} is a {self._model.task!r} model; a detection model is required"
            )

        self.task = "pose" if self._model.task == "pose" else "detect"
        if config.labels_file:
            self._labels = load_labels(config.labels_file)
        else:
            self._labels = {int(k): str(v) for k, v in self._model.names.items()}

    @property
    def labels(self) -> dict[int, str]:
        return self._labels

    def predict(
        self,
        image: Image.Image,
        *,
        min_confidence: float,
        class_ids: Collection[int] | None = None,
    ) -> list[Detection]:
        settings = self.config
        results = self._model.predict(
            image,
            conf=min_confidence,
            iou=settings.iou_threshold,
            imgsz=settings.image_size,
            max_det=settings.max_detections,
            classes=sorted(class_ids) if class_ids is not None else None,
            device=settings.device,
            verbose=False,
            # Only passed when enabled: recent Ultralytics versions deprecate the flag.
            **({"half": True} if settings.half else {}),
        )
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        keypoints_per_object = _keypoints_per_object(result.keypoints, len(result.boxes))
        detections = []
        for corners, score, class_index, keypoints in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
            keypoints_per_object,
            strict=True,
        ):
            class_id = int(class_index)
            detections.append(
                Detection(
                    label=self._labels.get(class_id, str(class_id)),
                    class_id=class_id,
                    confidence=float(score),
                    box=tuple(float(value) for value in corners),
                    keypoints=keypoints,
                )
            )
        return detections


Keypoints = tuple[tuple[float, float, float], ...]


def _keypoints_per_object(keypoints, object_count: int) -> list[Keypoints | None]:
    """Ultralytics keypoints as one tuple of (x, y, visibility) per detected object.

    Detection-only models have no keypoints: the list then holds ``None`` per object.
    """
    if keypoints is None or keypoints.xy is None:
        return [None] * object_count
    positions = keypoints.xy.cpu().tolist()
    visibilities = keypoints.conf.cpu().tolist() if keypoints.conf is not None else None
    return [
        tuple(
            (float(x), float(y), float(visibilities[obj][point]) if visibilities else 1.0)
            for point, (x, y) in enumerate(object_positions)
        )
        for obj, object_positions in enumerate(positions)
    ]
