"""The lab's resources: datasets, background jobs and model weights on disk."""

from __future__ import annotations

import threading
from pathlib import Path

from app.config import Settings
from app.lab.datasets import DatasetStore
from app.lab.jobs import JobManager

JOB_REFERENCE_PREFIX = "job:"


class WeightsNotFoundError(LookupError):
    """The requested weights are neither a file, a known model name nor a finished job."""


class LabState:
    """Everything the lab keeps under ``DETECT_DATA_DIR``."""

    def __init__(self, settings: Settings) -> None:
        self.data_dir = Path(settings.data_dir)
        self.weights_dir = self.data_dir / "weights"
        self.datasets = DatasetStore(self.data_dir / "datasets", settings.max_image_pixels)
        self.jobs = JobManager(self.data_dir / "jobs", self.weights_dir)
        # Loading a model takes seconds and a lot of memory: one at a time.
        self.model_switch_lock = threading.Lock()

    def resolve_weights(self, reference: str, settings: Settings) -> str:
        """Turn a weights reference into a path that the backend can load.

        Accepted references: ``job:<id>`` (the best weights of a training job), an
        existing path, or a bare model name such as ``yolo11s.pt`` (looked up in the
        weights folders; Ultralytics downloads known names to the returned path).
        """
        if reference.startswith(JOB_REFERENCE_PREFIX):
            job = self.jobs.get(reference.removeprefix(JOB_REFERENCE_PREFIX))
            trained_weights = (job.get("summary") or {}).get("weights")
            if not trained_weights:
                raise WeightsNotFoundError(f"job {job['id']} has no trained weights")
            return trained_weights
        if Path(reference).exists():
            return reference
        if "/" in reference:
            raise WeightsNotFoundError(f"weights {reference!r} not found")
        for folder in (self.weights_dir, Path(settings.model_weights).parent):
            candidate = folder / reference
            if candidate.exists():
                return str(candidate)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        return str(self.weights_dir / reference)

    def list_weight_files(self, settings: Settings) -> list[str]:
        """Weight files and exported models found in the weights folders."""
        model_suffixes = {".pt", ".onnx", ".engine", ".torchscript"}
        folders = {self.weights_dir, Path(settings.model_weights).parent}
        files = [
            str(path)
            for folder in folders
            if folder.is_dir()
            for path in sorted(folder.iterdir())
            if path.suffix in model_suffixes or path.name.endswith("_openvino_model")
        ]
        return sorted(set(files))
