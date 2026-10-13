"""Service configuration, read from environment variables prefixed with ``DETECT_``."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DETECT_",
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),
    )

    # --- Model selection -------------------------------------------------
    # Built-in name ("ultralytics", "torchvision") or a custom detector given
    # as "package.module:ClassName".
    backend: str = "ultralytics"
    # Weights to load. For ultralytics: a .pt/.onnx/... file or a known model
    # name (e.g. "yolo11n.pt", "best.pt" from a fine-tuning run). For
    # torchvision: a fine-tuned state_dict path, or a pretrained weights enum
    # member such as "DEFAULT".
    model_weights: str = "yolo11n.pt"
    # Architecture hint. ultralytics: "yolo" (default) or "rtdetr".
    # torchvision: a detection model name, e.g. "fasterrcnn_resnet50_fpn_v2".
    model_arch: str | None = None
    # Optional label file (one class name per line, line N = class id N).
    # Required for fine-tuned torchvision checkpoints.
    model_labels: str | None = None
    # Free-form tag echoed in responses to trace which model answered.
    model_version: str | None = None
    # "cpu", "cuda", "cuda:0", "mps"... None lets the backend decide.
    device: str | None = None

    # --- Detection defaults ----------------------------------------------
    # Classes reported when the request does not ask for specific ones.
    # Comma separated in the environment, e.g. DETECT_TARGET_CLASSES=cat,dog.
    # Use "*" to report every class the model knows.
    target_classes: Annotated[list[str], NoDecode] = ["cat"]
    confidence_threshold: float = Field(0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(0.45, ge=0.0, le=1.0)
    image_size: int = Field(640, gt=0)
    max_detections: int = Field(100, gt=0)

    # --- Input limits ------------------------------------------------------
    max_upload_bytes: int = Field(10 * 1024 * 1024, gt=0)
    max_image_pixels: int = Field(40_000_000, gt=0)

    # Run one inference on a blank image at startup so the first request
    # does not pay for lazy initialisation.
    warmup: bool = True

    @field_validator("target_classes", mode="before")
    @classmethod
    def _split_classes(cls, value: object) -> object:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
