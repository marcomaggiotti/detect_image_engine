"""Live inference settings, GPU information and model selection."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from app.lab.api.dependencies import CurrentSettings, Lab, Service
from app.lab.api.requests import ModelSwitch, RuntimeUpdate
from app.lab.model_switch import build_settings_for_weights, load_detection_service
from app.lab.options import BASE_MODELS
from app.service import DetectionService

router = APIRouter()


@router.get("/system")
def system_info(lab: Lab) -> dict:
    """PyTorch version, CUDA devices with their free memory, and the running job."""
    return {**_gpu_info(), "running_job": lab.jobs.running()}


def _gpu_info() -> dict:
    try:
        import torch
    except ImportError:  # torch is optional for the API
        return {"cuda": False, "devices": []}
    devices = []
    for index in range(torch.cuda.device_count()):
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        devices.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "memory_total_mb": total_bytes // 2**20,
                "memory_free_mb": free_bytes // 2**20,
            }
        )
    return {"torch": torch.__version__, "cuda": torch.cuda.is_available(), "devices": devices}


def _runtime_settings(service: DetectionService) -> dict:
    config = service.detector.config
    return {
        "confidence_threshold": service.confidence_threshold,
        "iou_threshold": config.iou_threshold,
        "image_size": config.image_size,
        "max_detections": config.max_detections,
        "half": config.half,
        "device": config.device,
        "default_classes": service.default_class_filter.class_names,
    }


@router.get("/runtime")
def get_runtime(service: Service) -> dict:
    return _runtime_settings(service)


@router.put("/runtime")
def update_runtime(update: RuntimeUpdate, service: Service) -> dict:
    """Change thresholds, NMS IoU, input size, max detections, FP16 or default classes live."""
    detector_settings = update.model_dump(
        include={"iou_threshold", "image_size", "max_detections", "half"}, exclude_none=True
    )
    default_classes = (
        update.default_classes or [] if "default_classes" in update.model_fields_set else None
    )
    service.reconfigure(
        confidence_threshold=update.confidence_threshold,
        default_classes=default_classes,
        **detector_settings,
    )
    return _runtime_settings(service)


@router.get("/weights")
def list_weights(lab: Lab, service: Service, settings: CurrentSettings) -> dict:
    """Pretrained choices, weight files on disk and the results of training jobs."""
    local_files = lab.list_weight_files(settings)
    trained = []
    for job in lab.jobs.list():
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
            local_files.append(summary["exported"])
    return {
        "active": {
            "weights": settings.model_weights,
            "arch": settings.model_arch,
            "task": service.detector.task,
        },
        "pretrained": BASE_MODELS,
        "local": sorted(set(local_files)),
        "trained": trained,
    }


@router.put("/model")
async def switch_model(
    request: Request, switch: ModelSwitch, lab: Lab, service: Service, settings: CurrentSettings
) -> dict:
    """Serve other weights: pretrained, fine-tuned (``job:<id>``) or exported."""
    weights_path = lab.resolve_weights(switch.weights, settings)
    new_settings = build_settings_for_weights(
        settings,
        weights_path,
        architecture=switch.arch,
        backend=switch.backend,
        version=switch.version,
    )

    def load() -> DetectionService:
        with lab.model_switch_lock:
            return load_detection_service(new_settings, service, switch.default_classes)

    try:
        new_service = await run_in_threadpool(load)
    except Exception as error:  # noqa: BLE001 - any loading failure is reported to the UI
        raise HTTPException(400, f"could not load {switch.weights}: {error}") from error
    request.app.state.service = new_service
    request.app.state.settings = new_settings
    return {
        "weights": weights_path,
        "task": new_service.detector.task,
        "labels": new_service.detector.labels,
        "default_classes": new_service.default_class_filter.class_names,
    }
