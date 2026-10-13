from __future__ import annotations

from collections.abc import Collection

from PIL import Image

from app.detectors import Detection, Detector, DetectorConfig


class FakeDetector(Detector):
    """Returns a fixed scene: one person, two cats and a low-confidence dog."""

    backend_name = "fake"
    SCENE = [
        Detection("person", 0, 0.92, (1.0, 2.0, 30.0, 60.0)),
        Detection("cat", 15, 0.88, (10.0, 10.0, 40.0, 40.0)),
        Detection("cat", 15, 0.41, (50.0, 5.0, 70.0, 25.0)),
        Detection("dog", 16, 0.20, (0.0, 0.0, 20.0, 20.0)),
    ]

    def __init__(self, config: DetectorConfig | None = None) -> None:
        super().__init__(config or DetectorConfig(weights="fake.pt"))
        self.calls: list[dict] = []

    @property
    def labels(self) -> dict[int, str]:
        return {0: "person", 15: "cat", 16: "dog", 56: "chair"}

    def detect(
        self,
        image: Image.Image,
        *,
        confidence: float,
        class_ids: Collection[int] | None = None,
    ) -> list[Detection]:
        self.calls.append({"size": image.size, "confidence": confidence, "class_ids": class_ids})
        return [
            d
            for d in self.SCENE
            if d.confidence >= confidence and (class_ids is None or d.class_id in class_ids)
        ]


class NotADetector:
    pass
