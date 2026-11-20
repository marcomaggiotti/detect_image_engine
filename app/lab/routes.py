"""HTTP API of the lab, mounted under ``/v1/lab``."""

from __future__ import annotations

import statistics
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Body, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.imaging import InvalidImageError, decode_image
from app.lab import robustness
from app.lab.datasets import DatasetError, DatasetStore, NotFound, thumbnail, to_normalised
from app.lab.jobs import JobError, JobManager
from app.lab.options import BASE_MODELS, nest, options
from app.service import DetectionService, UnknownClassError

router = APIRouter(prefix="/v1/lab", tags=["lab"])


class Lab:
    def __init__(self, settings: Settings) -> None:
        root = Path(settings.data_dir)
        self.root = root
        self.weights_dir = root / "weights"
        self.datasets = DatasetStore(root / "datasets", settings.max_image_pixels)
        self.jobs = JobManager(root / "jobs", self.weights_dir)
        self.model_lock = threading.Lock()

    def resolve_weights(self, weights: str, settings: Settings) -> str:
        if weights.startswith("job:"):
            job = self.jobs.get(weights[4:])
            summary = job.get("summary") or {}
            if "weights" not in summary:
                raise HTTPException(400, "that job has no trained weights")
            return summary["weights"]
        if Path(weights).exists():
            return weights
        if "/" not in weights:
            for folder in (self.weights_dir, Path(settings.model_weights).parent):
                if (folder / weights).exists():
                    return str(folder / weights)
            # Ultralytics downloads known model names to this path.
            self.weights_dir.mkdir(parents=True, exist_ok=True)
            return str(self.weights_dir / weights)
        raise HTTPException(400, f"weights {weights!r} not found")


def lab(request: Request) -> Lab:
    return request.app.state.lab


def service(request: Request) -> DetectionService:
    return request.app.state.service


def _errors(exc: Exception) -> HTTPException:
    if isinstance(exc, NotFound | LookupError):
        return HTTPException(404, str(exc))
    if isinstance(exc, JobError):
        return HTTPException(409, str(exc))
    return HTTPException(400, str(exc))


async def _read_image(upload: UploadFile, settings: Settings) -> Image.Image:
    data = await upload.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(413, "image too large")
    try:
        return decode_image(data, max_pixels=settings.max_image_pixels)
    except InvalidImageError as exc:
        raise HTTPException(400, str(exc)) from exc


# --- system, runtime settings and model selection ---------------------------
@router.get("/system")
def system(request: Request) -> dict:
    info: dict[str, Any] = {"cuda": False, "devices": []}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            info["devices"].append(
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_total_mb": total // 2**20,
                    "memory_free_mb": free // 2**20,
                }
            )
    except Exception:  # noqa: BLE001 - torch is optional for the API
        pass
    info["running_job"] = lab(request).jobs.running()
    return info


def _runtime(request: Request) -> dict:
    svc = service(request)
    cfg = svc.detector.config
    return {
        "confidence_threshold": svc.confidence_threshold,
        "iou_threshold": cfg.iou_threshold,
        "image_size": cfg.image_size,
        "max_detections": cfg.max_detections,
        "half": cfg.half,
        "device": cfg.device,
        "default_classes": svc.default_selection.names,
    }


@router.get("/runtime")
def get_runtime(request: Request) -> dict:
    return _runtime(request)


@router.put("/runtime")
def put_runtime(request: Request, body: Annotated[dict, Body()]) -> dict:
    """Change inference settings live: thresholds, NMS IoU, input size, FP16."""
    svc = service(request)
    updates = {}
    for key, field, cast, lo, hi in (
        ("iou_threshold", "iou_threshold", float, 0.0, 1.0),
        ("image_size", "image_size", int, 32, 4096),
        ("max_detections", "max_detections", int, 1, 3000),
    ):
        if body.get(key) is not None:
            value = cast(body[key])
            if not lo <= value <= hi:
                raise HTTPException(422, f"{key} must be between {lo} and {hi}")
            updates[field] = value
    if body.get("half") is not None:
        updates["half"] = bool(body["half"])
    with svc._lock:
        svc.detector.config = replace(svc.detector.config, **updates)
        if body.get("confidence_threshold") is not None:
            value = float(body["confidence_threshold"])
            if not 0 <= value <= 1:
                raise HTTPException(422, "confidence_threshold must be between 0 and 1")
            svc.confidence_threshold = value
        if "default_classes" in body:
            classes = body["default_classes"] or ["*"]
            try:
                svc.default_selection = svc.select_classes(list(classes))
            except UnknownClassError as exc:
                raise HTTPException(400, str(exc)) from exc
    return _runtime(request)


@router.get("/weights")
def weights(request: Request) -> dict:
    """Pretrained choices, local weight files and the results of training jobs."""
    the_lab = lab(request)
    local = []
    for folder in {the_lab.weights_dir, Path(request.app.state.settings.model_weights).parent}:
        if folder.is_dir():
            local += [
                str(p)
                for p in sorted(folder.iterdir())
                if p.suffix in {".pt", ".onnx", ".engine", ".torchscript"}
                or p.name.endswith("_openvino_model")
            ]
    trained = []
    for job in the_lab.jobs.list():
        summary = job.get("summary") or {}
        if summary.get("weights"):
            trained.append(
                {
                    "ref": f"job:{job['id']}",
                    "name": job.get("name"),
                    "weights": summary["weights"],
                    "exported": summary.get("exported"),
                    "metrics": summary.get("metrics"),
                    "task": summary.get("task"),
                }
            )
        elif summary.get("exported"):
            local.append(summary["exported"])
    return {
        "active": {
            "weights": request.app.state.settings.model_weights,
            "arch": request.app.state.settings.model_arch,
            "task": service(request).detector.task,
        },
        "pretrained": BASE_MODELS,
        "local": sorted(set(local)),
        "trained": trained,
    }


@router.put("/model")
async def switch_model(request: Request, body: Annotated[dict, Body()]) -> dict:
    """Load other weights (pretrained, fine-tuned ``job:<id>``, or exported)."""
    from app.main import build_detector

    the_lab = lab(request)
    settings: Settings = request.app.state.settings
    if not body.get("weights"):
        raise HTTPException(422, "weights is required")
    resolved = the_lab.resolve_weights(body["weights"], settings)
    new_settings = settings.model_copy(
        update={
            "model_weights": resolved,
            "model_arch": body.get("arch") or None,
            "backend": body.get("backend") or settings.backend,
            "model_version": body.get("version") or Path(resolved).stem,
            "model_labels": None,
        }
    )
    current = service(request)
    old_cfg = current.detector.config

    def load():
        with the_lab.model_lock:
            detector = build_detector(new_settings)
            detector.config = replace(
                detector.config,
                iou_threshold=old_cfg.iou_threshold,
                image_size=old_cfg.image_size,
                max_detections=old_cfg.max_detections,
                half=old_cfg.half,
            )
            wanted = body.get("default_classes") or current.default_selection.names or ["*"]
            try:
                svc = DetectionService(
                    detector,
                    default_classes=wanted,
                    confidence_threshold=current.confidence_threshold,
                    max_image_pixels=settings.max_image_pixels,
                )
            except UnknownClassError:
                svc = DetectionService(
                    detector,
                    default_classes=["*"],
                    confidence_threshold=current.confidence_threshold,
                    max_image_pixels=settings.max_image_pixels,
                )
            if settings.warmup:
                detector.warmup()
            return svc

    try:
        new_service = await run_in_threadpool(load)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - surface load errors to the UI
        raise HTTPException(400, f"could not load {body['weights']}: {exc}") from exc
    request.app.state.service = new_service
    request.app.state.settings = new_settings
    return {
        "weights": resolved,
        "task": new_service.detector.task,
        "labels": new_service.detector.labels,
        "default_classes": new_service.default_selection.names,
    }


# --- datasets & annotation ------------------------------------------------
@router.get("/datasets")
def list_datasets(request: Request) -> list[dict]:
    return lab(request).datasets.list()


@router.post("/datasets", status_code=201)
def create_dataset(request: Request, body: Annotated[dict, Body()]) -> dict:
    try:
        return lab(request).datasets.create(
            body.get("name", ""),
            body.get("classes") or [],
            body.get("task", "detect"),
            body.get("description", ""),
        )
    except DatasetError as exc:
        raise _errors(exc) from exc


@router.get("/datasets/{dataset_id}")
def get_dataset(request: Request, dataset_id: str) -> dict:
    store = lab(request).datasets
    try:
        return {**store.meta(dataset_id), "images": store.images(dataset_id)}
    except (NotFound, DatasetError) as exc:
        raise _errors(exc) from exc


@router.delete("/datasets/{dataset_id}", status_code=204)
def delete_dataset(request: Request, dataset_id: str) -> Response:
    try:
        lab(request).datasets.delete(dataset_id)
    except (NotFound, DatasetError) as exc:
        raise _errors(exc) from exc
    return Response(status_code=204)


@router.put("/datasets/{dataset_id}/classes")
def update_classes(request: Request, dataset_id: str, body: Annotated[dict, Body()]) -> dict:
    try:
        return lab(request).datasets.update_classes(dataset_id, body.get("classes") or [])
    except (NotFound, DatasetError) as exc:
        raise _errors(exc) from exc


@router.get("/datasets/{dataset_id}/stats")
def dataset_stats(
    request: Request, dataset_id: str, power: Annotated[float, Query(ge=0, le=2)] = 0.5
) -> dict:
    try:
        return lab(request).datasets.stats(dataset_id, power)
    except (NotFound, DatasetError) as exc:
        raise _errors(exc) from exc


@router.post("/datasets/{dataset_id}/images", status_code=201)
async def upload_images(
    request: Request, dataset_id: str, files: Annotated[list[UploadFile], File()]
) -> dict:
    store = lab(request).datasets
    settings: Settings = request.app.state.settings
    saved, errors = [], []
    for f in files:
        data = await f.read(settings.max_upload_bytes + 1)
        try:
            if len(data) > settings.max_upload_bytes:
                raise DatasetError("too large")
            saved.append(store.add_image(dataset_id, f.filename or "image.jpg", data))
        except NotFound as exc:
            raise _errors(exc) from exc
        except (DatasetError, InvalidImageError) as exc:
            errors.append({"file": f.filename, "error": str(exc)})
    return {"saved": saved, "errors": errors}


@router.get("/datasets/{dataset_id}/images/{filename}")
def get_image(request: Request, dataset_id: str, filename: str, thumb: bool = False):
    try:
        path = lab(request).datasets.image_file(dataset_id, filename)
    except (NotFound, DatasetError) as exc:
        raise _errors(exc) from exc
    if thumb:
        return Response(thumbnail(path), media_type="image/jpeg")
    return FileResponse(path)


@router.delete("/datasets/{dataset_id}/images/{filename}", status_code=204)
def delete_image(request: Request, dataset_id: str, filename: str) -> Response:
    try:
        lab(request).datasets.delete_image(dataset_id, filename)
    except (NotFound, DatasetError) as exc:
        raise _errors(exc) from exc
    return Response(status_code=204)


@router.get("/datasets/{dataset_id}/labels/{filename}")
def get_labels(request: Request, dataset_id: str, filename: str) -> dict:
    try:
        boxes = lab(request).datasets.labels(dataset_id, filename)
    except (NotFound, DatasetError) as exc:
        raise _errors(exc) from exc
    return {"annotated": boxes is not None, "boxes": boxes or []}


@router.put("/datasets/{dataset_id}/labels/{filename}")
def put_labels(request: Request, dataset_id: str, filename: str, body: Annotated[dict, Body()]):
    try:
        n = lab(request).datasets.save_labels(dataset_id, filename, body.get("boxes") or [])
    except (NotFound, DatasetError, KeyError, ValueError) as exc:
        raise _errors(exc) from exc
    return {"saved": n}


@router.delete("/datasets/{dataset_id}/labels/{filename}", status_code=204)
def delete_labels(request: Request, dataset_id: str, filename: str) -> Response:
    try:
        lab(request).datasets.clear_labels(dataset_id, filename)
    except (NotFound, DatasetError) as exc:
        raise _errors(exc) from exc
    return Response(status_code=204)


@router.post("/datasets/{dataset_id}/autolabel")
async def autolabel(request: Request, dataset_id: str, body: Annotated[dict, Body()] = None):
    """Pre-label images with the active model (classes matched by name);
    correcting boxes is much faster than drawing them."""
    body = body or {}
    store = lab(request).datasets
    svc = service(request)
    try:
        meta = store.meta(dataset_id)
        images = store.images(dataset_id)
    except (NotFound, DatasetError) as exc:
        raise _errors(exc) from exc
    ids = {name.lower(): i for i, name in enumerate(meta["classes"])}
    usable = [
        n for n in meta["classes"] if n.lower() in {v.lower() for v in svc.detector.labels.values()}
    ]
    if not usable:
        raise HTTPException(400, "the active model knows none of this dataset's classes")
    only = body.get("files")
    overwrite = bool(body.get("overwrite"))
    conf = float(body.get("confidence", 0.4))

    def run() -> dict:
        labelled = 0
        for item in images:
            if only and item["file"] not in only:
                continue
            if item["annotated"] and not overwrite:
                continue
            path = store.image_file(dataset_id, item["file"])
            with Image.open(path) as im:
                image = im.convert("RGB")
            result = svc.detect_image(image, classes=usable, confidence=conf)
            boxes = []
            for d in result.detections:
                box = {
                    "cls": ids[d.label.lower()],
                    **to_normalised(d.box, image.width, image.height),
                }
                if d.keypoints:
                    box["keypoints"] = [
                        [
                            x / image.width,
                            y / image.height,
                            2.0 if v > 0.5 else (1.0 if v > 0 else 0.0),
                        ]
                        for x, y, v in d.keypoints
                    ]
                boxes.append(box)
            store.save_labels(dataset_id, item["file"], boxes)
            labelled += 1
        return {"labelled": labelled, "classes": usable}

    return await run_in_threadpool(run)


@router.post("/datasets/{dataset_id}/import")
async def import_zip(request: Request, dataset_id: str, archive: Annotated[UploadFile, File()]):
    try:
        return lab(request).datasets.import_zip(dataset_id, await archive.read())
    except (NotFound, DatasetError) as exc:
        raise _errors(exc) from exc


@router.get("/datasets/{dataset_id}/export")
def export_zip(request: Request, dataset_id: str) -> Response:
    try:
        data = lab(request).datasets.export_zip(dataset_id)
    except (NotFound, DatasetError) as exc:
        raise _errors(exc) from exc
    return Response(
        data,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{dataset_id}.zip"'},
    )


# --- training jobs -----------------------------------------------------------
@router.get("/options")
def get_options() -> dict:
    return options()


@router.get("/jobs")
def list_jobs(request: Request) -> list[dict]:
    return lab(request).jobs.list()


@router.post("/jobs", status_code=201)
def start_job(request: Request, body: Annotated[dict, Body()]) -> dict:
    """Start a job. ``options`` may be flat ("loss.box": 7.5) or nested."""
    the_lab = lab(request)
    config = {**body, **nest(body.pop("options", None) or {})}
    kind = config.setdefault("kind", "train")
    try:
        if kind in {"train", "val"}:
            dataset_id = config.get("dataset")
            if not dataset_id:
                raise HTTPException(422, "dataset is required")
            config["dataset_dir"] = str(the_lab.datasets.path(dataset_id).resolve())
        if kind in {"export", "val"}:
            config["weights"] = the_lab.resolve_weights(
                config.get("weights") or request.app.state.settings.model_weights,
                request.app.state.settings,
            )
            if kind == "export" and not config.get("format"):
                raise HTTPException(422, "format is required (onnx, engine, openvino...)")
        if kind == "train" and config.get("base"):
            base = config["base"]
            if base.startswith("job:"):
                config["base"] = the_lab.resolve_weights(base, request.app.state.settings)
        return the_lab.jobs.start(config, name=config.get("name"))
    except HTTPException:
        raise
    except (NotFound, DatasetError, JobError, LookupError) as exc:
        raise _errors(exc) from exc


@router.get("/jobs/{job_id}")
def get_job(request: Request, job_id: str) -> dict:
    try:
        return lab(request).jobs.get(job_id)
    except LookupError as exc:
        raise _errors(exc) from exc


@router.post("/jobs/{job_id}/cancel")
def cancel_job(request: Request, job_id: str) -> dict:
    try:
        return lab(request).jobs.cancel(job_id)
    except (LookupError, JobError) as exc:
        raise _errors(exc) from exc


@router.get("/jobs/{job_id}/log")
def job_log(request: Request, job_id: str, tail: int = 200) -> dict:
    try:
        return {"log": lab(request).jobs.log(job_id, tail)}
    except LookupError as exc:
        raise _errors(exc) from exc


@router.get("/jobs/{job_id}/results")
def job_results(request: Request, job_id: str) -> dict:
    try:
        jobs = lab(request).jobs
        return {"columns": jobs.results(job_id), "artifacts": jobs.artifacts(job_id)}
    except LookupError as exc:
        raise _errors(exc) from exc


@router.get("/jobs/{job_id}/artifacts/{name}")
def job_artifact(request: Request, job_id: str, name: str):
    try:
        return FileResponse(lab(request).jobs.artifact(job_id, name))
    except LookupError as exc:
        raise _errors(exc) from exc


# --- robustness & latency ---------------------------------------------------
@router.get("/robustness/perturbations")
def perturbations() -> list[str]:
    return list(robustness.PERTURBATIONS)


@router.post("/robustness")
async def robustness_check(
    request: Request,
    image: Annotated[UploadFile, File()],
    perturbations: Annotated[
        str, Form()
    ] = "darken,gaussian_blur,motion_blur,noise,occlusion,hflip",
    severities: Annotated[str, Form()] = "1,3,5",
    classes: Annotated[str | None, Form()] = None,
    confidence: Annotated[float | None, Form()] = None,
) -> dict:
    settings: Settings = request.app.state.settings
    picture = await _read_image(image, settings)
    svc = service(request)
    names = [c.strip() for c in classes.split(",") if c.strip()] if classes else None
    try:
        svc.select_classes(names)
    except UnknownClassError as exc:
        raise HTTPException(400, str(exc)) from exc

    def detect(im: Image.Image):
        return svc.detect_image(im, classes=names, confidence=confidence).detections

    try:
        return await run_in_threadpool(
            robustness.evaluate,
            picture,
            detect,
            [p.strip() for p in perturbations.split(",") if p.strip()],
            [int(s) for s in severities.split(",") if s.strip()],
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/benchmark")
async def benchmark(
    request: Request,
    image: Annotated[UploadFile | None, File()] = None,
    sizes: Annotated[str, Form()] = "320,480,640",
    runs: Annotated[int, Form(ge=1, le=500)] = 20,
    half: Annotated[str, Form()] = "false,true",
) -> dict:
    """Latency of the active model at several input sizes, FP32 vs FP16."""
    settings: Settings = request.app.state.settings
    picture = (
        await _read_image(image, settings)
        if image is not None
        else Image.new("RGB", (1280, 720), (128, 128, 128))
    )
    svc = service(request)
    size_list = [int(s) for s in sizes.split(",") if s.strip()]
    half_list = [h.strip().lower() == "true" for h in half.split(",") if h.strip()]

    def run() -> dict:
        rows = []
        with svc._lock:
            original = svc.detector.config
            try:
                for h in half_list:
                    for size in size_list:
                        svc.detector.config = replace(original, image_size=size, half=h)
                        try:
                            svc.detector.detect(picture, confidence=0.25)  # warm-up
                            times, found = [], 0
                            for _ in range(runs):
                                start = time.perf_counter()
                                found = len(svc.detector.detect(picture, confidence=0.25))
                                times.append((time.perf_counter() - start) * 1000)
                        except Exception as exc:  # noqa: BLE001 - e.g. FP16 on CPU
                            rows.append({"image_size": size, "half": h, "error": str(exc)})
                            continue
                        times.sort()
                        rows.append(
                            {
                                "image_size": size,
                                "half": h,
                                "mean_ms": round(statistics.fmean(times), 2),
                                "p50_ms": round(times[len(times) // 2], 2),
                                "p95_ms": round(
                                    times[min(len(times) - 1, int(len(times) * 0.95))], 2
                                ),
                                "fps": round(1000 / statistics.fmean(times), 1),
                                "detections": found,
                            }
                        )
            finally:
                svc.detector.config = original
        return {
            "weights": settings.model_weights,
            "device": svc.detector.config.device,
            "runs": runs,
            "results": rows,
        }

    return await run_in_threadpool(run)
