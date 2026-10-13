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

_BOX_TASKS = {"detect", "segment"}


class UltralyticsDetector(Detector):
    backend_name = "ultralytics"

    def __init__(self, config: DetectorConfig) -> None:
        super().__init__(config)
        # Never let the library pip-install packages inside a running service.
        os.environ.setdefault("YOLO_AUTOINSTALL", "false")
        from ultralytics import RTDETR, YOLO

        arch = (config.arch or "yolo").lower()
        if arch == "rtdetr":
            self._model = RTDETR(config.weights)
        elif arch == "yolo":
            self._model = YOLO(config.weights)
        else:
            raise ValueError(
                f"Unsupported ultralytics arch {config.arch!r} (use 'yolo' or 'rtdetr')"
            )

        if self._model.task not in _BOX_TASKS:
            raise ValueError(
                f"{config.weights!r} is a {self._model.task!r} model; a detection model is required"
            )

        if config.labels_file:
            self._labels = load_labels(config.labels_file)
        else:
            self._labels = {int(k): str(v) for k, v in self._model.names.items()}

    @property
    def labels(self) -> dict[int, str]:
        return self._labels

    def detect(
        self,
        image: Image.Image,
        *,
        confidence: float,
        class_ids: Collection[int] | None = None,
    ) -> list[Detection]:
        cfg = self.config
        results = self._model.predict(
            image,
            conf=confidence,
            iou=cfg.iou_threshold,
            imgsz=cfg.image_size,
            max_det=cfg.max_detections,
            classes=sorted(class_ids) if class_ids is not None else None,
            device=cfg.device,
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        detections = []
        for xyxy, score, cls in zip(
            boxes.xyxy.cpu().tolist(),
            boxes.conf.cpu().tolist(),
            boxes.cls.cpu().tolist(),
            strict=True,
        ):
            class_id = int(cls)
            detections.append(
                Detection(
                    label=self._labels.get(class_id, str(class_id)),
                    class_id=class_id,
                    confidence=float(score),
                    box=tuple(float(v) for v in xyxy),
                )
            )
        return detections
