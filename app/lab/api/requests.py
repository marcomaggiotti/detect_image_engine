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


class SearchParameter(BaseModel):
    """One hyperparameter to search: listed ``values`` (grid or random), or a range."""

    key: str = Field(
        min_length=1, description='A training option key, e.g. "lr0" or "augment.mosaic"'
    )
    values: list[Any] | None = None
    min: float | None = None
    max: float | None = None
    log: bool = False  # draw on a log scale (learning rates)


class TuneSearch(BaseModel):
    strategy: Literal["grid", "random"] = "grid"
    parameters: list[SearchParameter] = Field(min_length=1, max_length=6)
    trials: int = Field(8, ge=1, le=30)
    trial_epochs: int = Field(15, ge=1, le=300)
    objective: Literal["pose.map50_95", "pose.map50", "box.map50_95", "box.map50"] = "box.map50_95"


class JobRequest(BaseModel):
    """A training, validation, export or tuning (hyperparameter search) job.

    Training options (``options``, flat ``"loss.box": 7.5`` or nested) and any other
    field are passed to ``training/run_job.py`` unchanged.
    """

    model_config = ConfigDict(extra="allow")

    kind: Literal["train", "val", "export", "tune"] = "train"
    name: str | None = None
    dataset: str | None = None
    weights: str | None = None
    base: str | None = None
    format: str | None = None
    options: dict[str, Any] | None = None
    search: TuneSearch | None = None  # kind "tune": the hyperparameter search
