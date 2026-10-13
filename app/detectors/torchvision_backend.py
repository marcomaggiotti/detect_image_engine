"""torchvision backend: Faster R-CNN, RetinaNet, FCOS, SSD...

Two ways to load a model:

* Pretrained COCO weights: ``DETECT_MODEL_WEIGHTS`` is a weights enum member
  of the architecture, e.g. ``DEFAULT`` or ``COCO_V1``.
* Fine-tuned checkpoint: ``DETECT_MODEL_WEIGHTS`` is the path of a saved
  ``state_dict`` (optionally wrapped as ``{"model": state_dict}``) and
  ``DETECT_MODEL_LABELS`` lists the classes, line 0 being the background
  class (as torchvision detection models expect).
"""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path

from PIL import Image

from app.detectors.base import Detection, Detector, DetectorConfig, load_labels

DEFAULT_ARCH = "fasterrcnn_resnet50_fpn_v2"
_IGNORED_LABELS = {"__background__", "N/A"}


class TorchvisionDetector(Detector):
    backend_name = "torchvision"

    def __init__(self, config: DetectorConfig) -> None:
        super().__init__(config)
        import torch
        import torchvision

        self._torch = torch
        arch = config.arch or DEFAULT_ARCH

        if Path(config.weights).is_file():
            if not config.labels_file:
                raise ValueError("A fine-tuned torchvision checkpoint needs DETECT_MODEL_LABELS")
            all_labels = load_labels(config.labels_file)
            model = torchvision.models.get_model(
                arch, weights=None, weights_backbone=None, num_classes=max(all_labels) + 1
            )
            state = torch.load(config.weights, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            model.load_state_dict(state)
        else:
            weights_enum = torchvision.models.get_model_weights(arch)
            try:
                weights = weights_enum[config.weights]
            except KeyError:
                members = ", ".join(weights_enum.__members__)
                raise ValueError(
                    f"{config.weights!r} is neither a file nor a {arch} weights name ({members})"
                ) from None
            model = torchvision.models.get_model(arch, weights=weights)
            all_labels = (
                load_labels(config.labels_file)
                if config.labels_file
                else dict(enumerate(weights.meta["categories"]))
            )

        self._labels = {i: n for i, n in all_labels.items() if n not in _IGNORED_LABELS}
        self._device = torch.device(
            config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model = model.eval().to(self._device)

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
        from torchvision.transforms.functional import to_tensor

        tensor = to_tensor(image).to(self._device)
        with self._torch.inference_mode():
            output = self._model([tensor])[0]

        detections = []
        for xyxy, score, cls in zip(
            output["boxes"].cpu().tolist(),
            output["scores"].cpu().tolist(),
            output["labels"].cpu().tolist(),
            strict=True,
        ):
            class_id = int(cls)
            if score < confidence or class_id not in self._labels:
                continue
            if class_ids is not None and class_id not in class_ids:
                continue
            detections.append(
                Detection(
                    label=self._labels[class_id],
                    class_id=class_id,
                    confidence=float(score),
                    box=tuple(float(v) for v in xyxy),
                )
            )
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections[: self.config.max_detections]
