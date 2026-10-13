"""Backend-agnostic detector interface.

Every model backend implements :class:`Detector`. The rest of the service only
talks to this interface, so switching model family (YOLO, RT-DETR, Faster
R-CNN, a custom model...) or loading fine-tuned weights is a configuration
change, not a code change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class Detection:
    label: str
    class_id: int
    confidence: float
    # Pixel coordinates in the input image: (x1, y1, x2, y2).
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class DetectorConfig:
    weights: str
    arch: str | None = None
    labels_file: str | None = None
    device: str | None = None
    iou_threshold: float = 0.45
    image_size: int = 640
    max_detections: int = 100


class Detector(ABC):
    """An object detection model."""

    #: Short name reported by the API.
    backend_name: str = "custom"

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config

    @property
    @abstractmethod
    def labels(self) -> dict[int, str]:
        """Class id -> class name for every class the model can detect."""

    @abstractmethod
    def detect(
        self,
        image: Image.Image,
        *,
        confidence: float,
        class_ids: Collection[int] | None = None,
    ) -> list[Detection]:
        """Detect objects in an RGB image.

        Only detections with score >= ``confidence`` are returned. When
        ``class_ids`` is given, only those classes are returned.
        """

    def warmup(self) -> None:
        self.detect(Image.new("RGB", (64, 64)), confidence=1.0)


def load_labels(path: str | Path) -> dict[int, str]:
    """Read a label file: one class name per line, line index = class id."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return {i: name.strip() for i, name in enumerate(lines) if name.strip()}
