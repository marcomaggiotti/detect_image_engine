"""Detection use case: decode, choose classes, run the model, summarise."""

from __future__ import annotations

import threading
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from app.detectors import Detection, Detector
from app.imaging import decode_image

ALL_CLASSES = "*"


class UnknownClassError(ValueError):
    def __init__(self, unknown: list[str]) -> None:
        super().__init__(f"Unknown class(es) for this model: {', '.join(unknown)}")
        self.unknown = unknown


@dataclass(frozen=True)
class ClassSelection:
    """Class ids to report; ``ids is None`` means every class."""

    names: list[str] | None
    ids: frozenset[int] | None


@dataclass
class DetectionResult:
    width: int
    height: int
    classes: list[str] | None
    detections: list[Detection]
    inference_ms: float
    counts: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.counts = dict(Counter(d.label for d in self.detections))


class DetectionService:
    def __init__(
        self,
        detector: Detector,
        *,
        default_classes: Iterable[str],
        confidence_threshold: float,
        max_image_pixels: int,
    ) -> None:
        self.detector = detector
        self.confidence_threshold = confidence_threshold
        self.max_image_pixels = max_image_pixels
        self._ids_by_name: dict[str, set[int]] = {}
        for class_id, name in detector.labels.items():
            self._ids_by_name.setdefault(name.lower(), set()).add(class_id)
        # Validate at startup: a model that cannot detect the configured
        # target classes should fail fast, not on the first request.
        self.default_selection = self.select_classes(list(default_classes))
        # Most backends are not safe to call from several threads at once.
        self._lock = threading.Lock()

    def select_classes(self, names: list[str] | None) -> ClassSelection:
        if names is None:
            return self.default_selection
        if not names or ALL_CLASSES in names:
            return ClassSelection(names=None, ids=None)
        unknown = [n for n in names if n.lower() not in self._ids_by_name]
        if unknown:
            raise UnknownClassError(unknown)
        ids = frozenset().union(*(self._ids_by_name[n.lower()] for n in names))
        return ClassSelection(names=list(dict.fromkeys(names)), ids=ids)

    def detect(
        self,
        data: bytes,
        *,
        classes: list[str] | None = None,
        confidence: float | None = None,
    ) -> DetectionResult:
        selection = self.select_classes(classes)
        image = decode_image(data, max_pixels=self.max_image_pixels)
        threshold = self.confidence_threshold if confidence is None else confidence

        start = time.perf_counter()
        with self._lock:
            detections = self.detector.detect(image, confidence=threshold, class_ids=selection.ids)
        elapsed_ms = (time.perf_counter() - start) * 1000

        return DetectionResult(
            width=image.width,
            height=image.height,
            classes=selection.names,
            detections=detections,
            inference_ms=round(elapsed_ms, 2),
        )
