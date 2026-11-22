"""Detection use case: validate the input, choose the classes, run the model, summarise.

The service is independent of both the HTTP layer and the model family: it receives
bytes or a Pillow image and talks to the model only through the :class:`Detector`
interface.
"""

from __future__ import annotations

import threading
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace

from PIL import Image

from app.detectors import Detection, Detector
from app.imaging import decode_image

ALL_CLASSES = "*"


class UnknownClassError(ValueError):
    """Raised when a requested class name is not known by the loaded model."""

    def __init__(self, unknown_names: list[str]) -> None:
        super().__init__(f"Unknown class(es) for this model: {', '.join(unknown_names)}")
        self.unknown_names = unknown_names


@dataclass(frozen=True)
class ClassFilter:
    """The classes to report. ``class_ids is None`` means every class of the model."""

    class_names: list[str] | None
    class_ids: frozenset[int] | None


@dataclass
class DetectionResult:
    image_width: int
    image_height: int
    class_names: list[str] | None
    detections: list[Detection]
    inference_ms: float
    counts_by_label: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.counts_by_label = dict(Counter(detection.label for detection in self.detections))


class DetectionService:
    """Runs one detector safely and turns its output into a :class:`DetectionResult`."""

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
        self._class_ids_by_name = self._index_class_names(detector.labels)
        # Fail at startup, not on the first request, if the model cannot detect
        # the configured default classes.
        self.default_class_filter = self.build_class_filter(list(default_classes))
        # Most model backends are not safe to call from several threads at once.
        self._inference_lock = threading.Lock()

    @staticmethod
    def _index_class_names(labels: dict[int, str]) -> dict[str, set[int]]:
        class_ids_by_name: dict[str, set[int]] = {}
        for class_id, class_name in labels.items():
            class_ids_by_name.setdefault(class_name.lower(), set()).add(class_id)
        return class_ids_by_name

    def build_class_filter(self, class_names: list[str] | None) -> ClassFilter:
        """Translate class names into model class ids.

        ``None`` gives the default classes, an empty list or ``"*"`` every class.
        Names are case-insensitive; an unknown name raises :class:`UnknownClassError`.
        """
        if class_names is None:
            return self.default_class_filter
        if not class_names or ALL_CLASSES in class_names:
            return ClassFilter(class_names=None, class_ids=None)
        unknown_names = [
            name for name in class_names if name.lower() not in self._class_ids_by_name
        ]
        if unknown_names:
            raise UnknownClassError(unknown_names)
        class_ids = frozenset().union(
            *(self._class_ids_by_name[name.lower()] for name in class_names)
        )
        unique_names = list(dict.fromkeys(class_names))
        return ClassFilter(class_names=unique_names, class_ids=class_ids)

    def detect_in_upload(
        self,
        image_bytes: bytes,
        *,
        class_names: list[str] | None = None,
        min_confidence: float | None = None,
    ) -> DetectionResult:
        """Decode and validate uploaded bytes, then detect (see :meth:`detect_in_image`)."""
        image = decode_image(image_bytes, max_pixels=self.max_image_pixels)
        return self.detect_in_image(image, class_names=class_names, min_confidence=min_confidence)

    def detect_in_image(
        self,
        image: Image.Image,
        *,
        class_names: list[str] | None = None,
        min_confidence: float | None = None,
    ) -> DetectionResult:
        """Detect the requested classes in an RGB image.

        ``min_confidence=None`` uses the service's threshold (an explicit ``0.0`` is kept).
        """
        class_filter = self.build_class_filter(class_names)
        threshold = self.confidence_threshold if min_confidence is None else min_confidence

        started = time.perf_counter()
        with self._inference_lock:
            detections = self.detector.predict(
                image, min_confidence=threshold, class_ids=class_filter.class_ids
            )
        inference_ms = (time.perf_counter() - started) * 1000

        return DetectionResult(
            image_width=image.width,
            image_height=image.height,
            class_names=class_filter.class_names,
            detections=detections,
            inference_ms=round(inference_ms, 2),
        )

    def reconfigure(
        self,
        *,
        confidence_threshold: float | None = None,
        default_classes: list[str] | None = None,
        **detector_settings: object,
    ) -> None:
        """Change settings live, between two inferences.

        ``detector_settings`` are :class:`DetectorConfig` fields (iou_threshold,
        image_size, max_detections, half). ``default_classes=[]`` selects every class.
        """
        with self._inference_lock:
            if default_classes is not None:
                self.default_class_filter = self.build_class_filter(
                    default_classes or [ALL_CLASSES]
                )
            if confidence_threshold is not None:
                self.confidence_threshold = confidence_threshold
            if detector_settings:
                self.detector.config = replace(self.detector.config, **detector_settings)

    @contextmanager
    def exclusive_detector(self) -> Iterator[Detector]:
        """Use the detector directly (e.g. to change its settings or benchmark it)
        without another request running at the same time."""
        with self._inference_lock:
            yield self.detector
