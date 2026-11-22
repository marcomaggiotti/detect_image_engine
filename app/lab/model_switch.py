"""Replace the served model without restarting the service."""

from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.service import ALL_CLASSES, DetectionService, UnknownClassError


def build_settings_for_weights(
    settings: Settings,
    weights_path: str,
    *,
    architecture: str | None,
    backend: str | None,
    version: str | None,
) -> Settings:
    return settings.model_copy(
        update={
            "model_weights": weights_path,
            "model_arch": architecture,
            "backend": backend or settings.backend,
            "model_version": version or Path(weights_path).stem,
            "model_labels": None,
        }
    )


def load_detection_service(
    new_settings: Settings,
    current_service: DetectionService,
    requested_default_classes: list[str] | None,
) -> DetectionService:
    """Load the new detector and wrap it in a service that keeps the current
    thresholds, input size and precision.

    The default classes are kept when the new model knows them, otherwise every
    class is reported (e.g. switching from a COCO model to a fine-tuned one).
    """
    from app.main import build_detector  # imported here: main imports the lab

    detector = build_detector(new_settings)
    current_config = current_service.detector.config
    new_service = _service_with_default_classes(
        detector,
        requested_default_classes
        or current_service.default_class_filter.class_names
        or [ALL_CLASSES],
        current_service.confidence_threshold,
        new_settings.max_image_pixels,
    )
    new_service.reconfigure(
        iou_threshold=current_config.iou_threshold,
        image_size=current_config.image_size,
        max_detections=current_config.max_detections,
        half=current_config.half,
    )
    if new_settings.warmup:
        detector.warmup()
    return new_service


def _service_with_default_classes(
    detector, default_classes: list[str], confidence_threshold: float, max_image_pixels: int
) -> DetectionService:
    try:
        return DetectionService(
            detector,
            default_classes=default_classes,
            confidence_threshold=confidence_threshold,
            max_image_pixels=max_image_pixels,
        )
    except UnknownClassError:
        return DetectionService(
            detector,
            default_classes=[ALL_CLASSES],
            confidence_threshold=confidence_threshold,
            max_image_pixels=max_image_pixels,
        )
