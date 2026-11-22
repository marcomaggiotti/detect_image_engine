"""Robustness under perturbations and latency benchmarks of the served model."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image
from starlette.concurrency import run_in_threadpool

from app.lab import robustness
from app.lab.api.dependencies import CurrentSettings, Service, read_uploaded_image, split_csv
from app.lab.benchmark import measure_latency

router = APIRouter()

DEFAULT_PERTURBATIONS = "darken,gaussian_blur,motion_blur,noise,occlusion,hflip"
BLANK_FRAME_SIZE = (1280, 720)


@router.get("/robustness/perturbations")
def list_perturbations() -> list[str]:
    return list(robustness.PERTURBATIONS)


@router.post("/robustness")
async def check_robustness(
    service: Service,
    settings: CurrentSettings,
    image: Annotated[UploadFile, File()],
    perturbations: Annotated[str, Form()] = DEFAULT_PERTURBATIONS,
    severities: Annotated[str, Form()] = "1,3,5",
    classes: Annotated[str | None, Form()] = None,
    confidence: Annotated[float | None, Form()] = None,
) -> dict:
    """Detections on perturbed copies of the picture compared with the clean picture."""
    picture = await read_uploaded_image(image, settings)
    class_names = split_csv(classes) or None
    service.build_class_filter(class_names)  # unknown classes: 400 before the long run

    def detect(perturbed: Image.Image):
        return service.detect_in_image(
            perturbed, class_names=class_names, min_confidence=confidence
        ).detections

    try:
        return await run_in_threadpool(
            robustness.evaluate,
            picture,
            detect,
            split_csv(perturbations),
            [int(severity) for severity in split_csv(severities)],
        )
    except ValueError as error:  # unknown perturbation name
        raise HTTPException(400, str(error)) from error


@router.post("/benchmark")
async def benchmark_latency(
    service: Service,
    settings: CurrentSettings,
    image: Annotated[UploadFile | None, File()] = None,
    sizes: Annotated[str, Form()] = "320,480,640",
    runs: Annotated[int, Form(ge=1, le=500)] = 20,
    half: Annotated[str, Form()] = "false,true",
) -> dict:
    """Latency of the active model at several input sizes, FP32 vs FP16."""
    picture = (
        await read_uploaded_image(image, settings)
        if image is not None
        else Image.new("RGB", BLANK_FRAME_SIZE, (128, 128, 128))
    )
    rows = await run_in_threadpool(
        measure_latency,
        service,
        picture,
        image_sizes=[int(size) for size in split_csv(sizes)],
        half_precision_options=[value.lower() == "true" for value in split_csv(half)],
        runs=runs,
    )
    return {
        "weights": settings.model_weights,
        "device": service.detector.config.device,
        "runs": runs,
        "results": rows,
    }
