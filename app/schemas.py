from __future__ import annotations

from pydantic import BaseModel, Field


class Box(BaseModel):
    """Bounding box in pixel coordinates of the uploaded image."""

    x1: float
    y1: float
    x2: float
    y2: float


class DetectionOut(BaseModel):
    label: str
    class_id: int
    confidence: float
    box: Box


class ImageInfo(BaseModel):
    width: int
    height: int


class ModelRef(BaseModel):
    backend: str
    weights: str
    version: str | None = None


class DetectResponse(BaseModel):
    model: ModelRef
    image: ImageInfo
    classes: list[str] | None = Field(
        description="Classes that were searched for; null means all model classes."
    )
    found: bool = Field(description="True when at least one requested object was detected.")
    counts: dict[str, int]
    detections: list[DetectionOut]
    inference_ms: float


class ModelInfo(ModelRef):
    arch: str | None = None
    default_classes: list[str] | None
    confidence_threshold: float
    labels: dict[int, str]


class ErrorResponse(BaseModel):
    detail: str
