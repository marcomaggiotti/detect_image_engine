"""Training-time customisation that Ultralytics' train() arguments don't expose.

* **Assignment strategy.** YOLO11/YOLOv8 match predictions to ground truth
  with a Task-Aligned Assigner: each object takes the ``topk`` anchors with
  the best alignment ``score**alpha * IoU**beta``. A higher ``beta`` favours
  localisation, a higher ``alpha`` favours classification, a larger ``topk``
  gives every object more positive samples (useful for small, dense objects).
  RT-DETR uses a Hungarian matcher whose cost weights (class / bbox / giou)
  can be changed the same way.
* **Classification loss for class imbalance.** The classification term is a
  per-class binary cross-entropy. :class:`WeightedFocalBCE` replaces it with
  a focal loss (``gamma`` down-weights easy, mostly background, anchors) and
  optional per-class weights (rarer classes count more).

Both are installed through Ultralytics callbacks, so they apply to the model
being trained and to its EMA copy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
class LossCustomization:
    tal_topk: int | None = None
    tal_alpha: float | None = None
    tal_beta: float | None = None
    # RT-DETR Hungarian matcher cost weights.
    match_class: float | None = None
    match_bbox: float | None = None
    match_giou: float | None = None
    focal_gamma: float = 0.0
    focal_alpha: float | None = None
    class_weights: Sequence[float] | None = None

    @property
    def changes_bce(self) -> bool:
        return self.focal_gamma > 0 or self.focal_alpha is not None or bool(self.class_weights)


def configure_criterion(criterion, custom: LossCustomization) -> list[str]:
    """Apply ``custom`` to an Ultralytics loss object. Returns what changed."""
    changed = []
    assigner = getattr(criterion, "assigner", None)
    if assigner is not None:
        for attr, value in (
            ("topk", custom.tal_topk),
            ("alpha", custom.tal_alpha),
            ("beta", custom.tal_beta),
        ):
            if value is not None and hasattr(assigner, attr):
                setattr(assigner, attr, value)
                changed.append(f"assigner.{attr}={value}")
    matcher = getattr(criterion, "matcher", None)
    if matcher is not None and isinstance(getattr(matcher, "cost_gain", None), dict):
        for key, value in (
            ("class", custom.match_class),
            ("bbox", custom.match_bbox),
            ("giou", custom.match_giou),
        ):
            if value is not None:
                matcher.cost_gain[key] = value
                changed.append(f"matcher.{key}={value}")
    if custom.changes_bce and hasattr(criterion, "bce"):
        from training.focal import WeightedFocalBCE

        criterion.bce = WeightedFocalBCE(
            custom.focal_gamma, custom.focal_alpha, custom.class_weights
        )
        changed.append(
            f"cls loss: focal gamma={custom.focal_gamma}, weights={custom.class_weights}"
        )
    return changed


def _customize_model(model, custom: LossCustomization, log) -> None:
    """Build the model's loss now (Ultralytics builds it lazily on the first
    batch) and customise it; ``model.loss()`` then reuses it."""
    if model is None or not hasattr(model, "init_criterion"):
        return
    if getattr(model, "criterion", None) is None:
        model.criterion = model.init_criterion()
    changed = configure_criterion(model.criterion, custom)
    if changed:
        log("Loss customisation: " + ", ".join(changed))


def install(yolo_model, custom: LossCustomization, log=print) -> None:
    """Register callbacks on an ``ultralytics.YOLO``/``RTDETR`` object."""

    def on_train_start(trainer):
        model = trainer.model
        _customize_model(getattr(model, "module", model), custom, log)  # unwrap DDP
        _customize_model(getattr(getattr(trainer, "ema", None), "ema", None), custom, log)

    yolo_model.add_callback("on_train_start", on_train_start)
