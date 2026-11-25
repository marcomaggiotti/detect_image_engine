"""Training, validation and export jobs."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.lab.api.dependencies import CurrentSettings, Lab
from app.lab.api.requests import JobRequest, TuneSearch
from app.lab.options import nest, options, tunable_options
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
    if request.kind == "tune":
        if request.search is None:
            raise HTTPException(422, "search is required for a tuning job")
        config["search"] = checked_search(request.search)
    if request.kind in {"train", "val", "tune"}:
        if not request.dataset:
            raise HTTPException(422, "dataset is required")
        config["dataset_dir"] = str(lab.datasets.path(request.dataset).resolve())
    if request.kind in {"export", "val"}:
        config["weights"] = lab.resolve_weights(request.weights or settings.model_weights, settings)
    if request.kind == "export" and not request.format:
        raise HTTPException(422, "format is required (onnx, engine, openvino...)")
    if (
        request.kind in {"train", "tune"}
        and request.base
        and request.base.startswith(JOB_REFERENCE_PREFIX)
    ):
        config["base"] = lab.resolve_weights(request.base, settings)
    return lab.jobs.start(config, name=request.name)


def _is_number(value) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _is_integer(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _outside(values: list, low: float | None, high: float | None) -> list:
    return [
        value
        for value in values
        if _is_number(value)
        and ((low is not None and value < low) or (high is not None and value > high))
    ]


def checked_search(search: TuneSearch) -> dict:
    """The search with each parameter checked against the training options (422 if wrong)."""
    known = tunable_options()
    parameters = []
    for parameter in search.parameters:
        option = known.get(parameter.key)
        if option is None:
            raise HTTPException(422, f"{parameter.key!r} is not a tunable training option")
        if parameter.values:
            values = list(dict.fromkeys(parameter.values))
            low, high = option.get("min"), option.get("max")
            if outside := _outside(values, low, high):
                raise HTTPException(422, f"{parameter.key}: {outside} outside [{low}, {high}]")
            if option.get("choices") and any(v not in option["choices"] for v in values):
                raise HTTPException(422, f"{parameter.key}: choose among {option['choices']}")
            parameters.append({"key": parameter.key, "values": values})
        elif (
            parameter.min is not None
            and parameter.max is not None
            and parameter.min < parameter.max
        ):
            if search.strategy == "grid":
                raise HTTPException(422, f"grid search: list the values of {parameter.key}")
            parameters.append({
                "key": parameter.key, "min": parameter.min, "max": parameter.max,
                "log": parameter.log, "integer": _is_integer(option.get("default")),
            })  # fmt: skip
        else:
            raise HTTPException(422, f"{parameter.key}: give values, or min < max")
    return {**search.model_dump(), "parameters": parameters}


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
