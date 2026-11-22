"""Request bodies of the lab API (validated by Pydantic: wrong types or ranges give 422)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class RuntimeUpdate(BaseModel):
    """Inference settings to change live; omitted fields are left unchanged."""

    confidence_threshold: float | None = Field(None, ge=0, le=1)
    iou_threshold: float | None = Field(None, ge=0, le=1)
    image_size: int | None = Field(None, ge=32, le=4096)
    max_detections: int | None = Field(None, ge=1, le=3000)
    half: bool | None = None
    # null or [] selects every class.
    default_classes: list[str] | None = None


class ModelSwitch(BaseModel):
    weights: str = Field(min_length=1, description="Model name, path or job:<id>")
    arch: str | None = None
    backend: str | None = None
    version: str | None = None
    default_classes: list[str] | None = None


class DatasetCreate(BaseModel):
    name: str
    classes: list[str]
    task: Literal["detect", "pose"] = "detect"
    description: str = ""


class DatasetClasses(BaseModel):
    classes: list[str]


class LabelBox(BaseModel):
    """One object in YOLO format: class index and centre/size normalised to 0-1."""

    cls: int = Field(ge=0)
    x: float
    y: float
    w: float
    h: float
    keypoints: list[list[float]] | None = None


class ImageLabels(BaseModel):
    boxes: list[LabelBox] = []


class PrelabelRequest(BaseModel):
    confidence: float = Field(0.4, ge=0, le=1)
    overwrite: bool = False
    files: list[str] | None = None


class JobRequest(BaseModel):
    """A training, validation or export job.

    Training options (``options``, flat ``"loss.box": 7.5`` or nested) and any other
    field are passed to ``training/run_job.py`` unchanged.
    """

    model_config = ConfigDict(extra="allow")

    kind: Literal["train", "val", "export"] = "train"
    name: str | None = None
    dataset: str | None = None
    weights: str | None = None
    base: str | None = None
    format: str | None = None
    options: dict[str, Any] | None = None
