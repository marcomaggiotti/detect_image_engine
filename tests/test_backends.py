"""Smoke tests of the real backends with randomly initialised models (no downloads)."""

from __future__ import annotations

import pytest
from PIL import Image

from app.detectors import DetectorConfig, create_detector

IMAGE = Image.new("RGB", (128, 96), (30, 60, 90))


def test_ultralytics_backend_from_architecture_yaml():
    pytest.importorskip("ultralytics")
    detector = create_detector("ultralytics", DetectorConfig(weights="yolo11n.yaml", device="cpu"))
    assert len(detector.labels) == 80  # untrained yaml models have numeric names
    detections = detector.detect(IMAGE, confidence=0.0, class_ids={15})
    assert all(d.class_id == 15 for d in detections)


def test_torchvision_backend_with_finetuned_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    torchvision = pytest.importorskip("torchvision")

    arch = "fasterrcnn_mobilenet_v3_large_320_fpn"
    model = torchvision.models.get_model(arch, weights=None, weights_backbone=None, num_classes=3)
    weights = tmp_path / "model.pt"
    torch.save({"model": model.state_dict()}, weights)
    labels = tmp_path / "labels.txt"
    labels.write_text("__background__\ncat\ndog\n")

    detector = create_detector(
        "torchvision",
        DetectorConfig(weights=str(weights), arch=arch, labels_file=str(labels), device="cpu"),
    )
    assert detector.labels == {1: "cat", 2: "dog"}
    detections = detector.detect(IMAGE, confidence=0.0, class_ids={1})
    assert all(d.label == "cat" for d in detections)


def test_torchvision_checkpoint_requires_labels(tmp_path):
    pytest.importorskip("torchvision")
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"")
    with pytest.raises(ValueError, match="DETECT_MODEL_LABELS"):
        create_detector("torchvision", DetectorConfig(weights=str(weights)))
