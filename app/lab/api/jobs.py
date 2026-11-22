"""Training, validation and export jobs."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.lab.api.dependencies import CurrentSettings, Lab
from app.lab.api.requests import JobRequest
from app.lab.options import nest, options
from app.lab.state import JOB_REFERENCE_PREFIX

router = APIRouter()


@router.get("/options")
def training_options() -> dict:
    """Every training option: default, range and meaning (the UI builds its form from it)."""
    return options()


@router.get("/jobs")
def list_jobs(lab: Lab) -> list[dict]:
    return lab.jobs.list()


@router.post("/jobs", status_code=201)
def start_job(request: JobRequest, lab: Lab, settings: CurrentSettings) -> dict:
    config = request.model_dump(exclude={"options"}, exclude_none=True)
    config.update(nest(request.options or {}))
    if request.kind in {"train", "val"}:
        if not request.dataset:
            raise HTTPException(422, "dataset is required")
        config["dataset_dir"] = str(lab.datasets.path(request.dataset).resolve())
    if request.kind in {"export", "val"}:
        config["weights"] = lab.resolve_weights(request.weights or settings.model_weights, settings)
    if request.kind == "export" and not request.format:
        raise HTTPException(422, "format is required (onnx, engine, openvino...)")
    if request.kind == "train" and request.base and request.base.startswith(JOB_REFERENCE_PREFIX):
        config["base"] = lab.resolve_weights(request.base, settings)
    return lab.jobs.start(config, name=request.name)


@router.get("/jobs/{job_id}")
def get_job(job_id: str, lab: Lab) -> dict:
    return lab.jobs.get(job_id)


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, lab: Lab) -> dict:
    return lab.jobs.cancel(job_id)


@router.get("/jobs/{job_id}/log")
def job_log(job_id: str, lab: Lab, tail: int = 200) -> dict:
    return {"log": lab.jobs.log(job_id, tail)}


@router.get("/jobs/{job_id}/results")
def job_results(job_id: str, lab: Lab) -> dict:
    """Per-epoch curves (results.csv) and the names of the plots."""
    return {"columns": lab.jobs.results(job_id), "artifacts": lab.jobs.artifacts(job_id)}


@router.get("/jobs/{job_id}/artifacts/{name}")
def job_artifact(job_id: str, name: str, lab: Lab) -> FileResponse:
    return FileResponse(lab.jobs.artifact(job_id, name))
