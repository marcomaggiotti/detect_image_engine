from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.detectors import Detection
from app.imaging import InvalidImageError
from app.schemas import (
    Box,
    DetectionOut,
    DetectResponse,
    ErrorResponse,
    ImageInfo,
    ModelInfo,
    ModelRef,
)
from app.service import DetectionService, UnknownClassError

router = APIRouter()


def _service(request: Request) -> DetectionService:
    return request.app.state.service


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _model_ref(request: Request) -> ModelRef:
    settings = _settings(request)
    return ModelRef(
        backend=_service(request).detector.backend_name,
        weights=settings.model_weights,
        version=settings.model_version,
    )


def _parse_class_names(query_values: list[str] | None) -> list[str] | None:
    """Accept both ?classes=cat,dog and ?classes=cat&classes=dog."""
    if not query_values:
        return None
    class_names = [
        name.strip() for value in query_values for name in value.split(",") if name.strip()
    ]
    return class_names or None


def _to_box(corners: tuple[float, float, float, float]) -> Box:
    x1, y1, x2, y2 = (round(value, 1) for value in corners)
    return Box(x1=x1, y1=y1, x2=x2, y2=y2)


def _to_detection_out(detection: Detection) -> DetectionOut:
    keypoints = (
        [[round(value, 2) for value in keypoint] for keypoint in detection.keypoints]
        if detection.keypoints
        else None
    )
    return DetectionOut(
        label=detection.label,
        class_id=detection.class_id,
        confidence=round(detection.confidence, 4),
        box=_to_box(detection.box),
        keypoints=keypoints,
    )


@router.get("/health", tags=["ops"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/v1/model", response_model=ModelInfo, tags=["detection"])
def model_info(request: Request) -> ModelInfo:
    service = _service(request)
    return ModelInfo(
        **_model_ref(request).model_dump(),
        arch=_settings(request).model_arch,
        task=service.detector.task,
        default_classes=service.default_class_filter.class_names,
        confidence_threshold=service.confidence_threshold,
        labels=service.detector.labels,
    )


@router.post(
    "/v1/detect",
    response_model=DetectResponse,
    tags=["detection"],
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image or unknown class"},
        413: {"model": ErrorResponse, "description": "Upload too large"},
    },
)
async def detect(
    request: Request,
    image: Annotated[UploadFile, File(description="Image file (JPEG, PNG, WebP, ...)")],
    classes: Annotated[
        list[str] | None,
        Query(
            description="Classes to detect, e.g. 'cat' or 'cat,dog,person'. "
            "'*' returns every class. Defaults to DETECT_TARGET_CLASSES."
        ),
    ] = None,
    confidence: Annotated[
        float | None,
        Query(ge=0.0, le=1.0, description="Minimum score. Defaults to the configured threshold."),
    ] = None,
) -> DetectResponse:
    service = _service(request)
    max_bytes = _settings(request).max_upload_bytes

    image_bytes = await image.read(max_bytes + 1)
    if len(image_bytes) > max_bytes:
        raise HTTPException(413, f"Image exceeds {max_bytes} bytes")

    try:
        result = await run_in_threadpool(
            service.detect_in_upload,
            image_bytes,
            class_names=_parse_class_names(classes),
            min_confidence=confidence,
        )
    except UnknownClassError as exc:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, f"{exc}. See GET /v1/model for available labels."
        ) from exc
    except InvalidImageError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    return DetectResponse(
        model=_model_ref(request),
        image=ImageInfo(width=result.image_width, height=result.image_height),
        classes=result.class_names,
        found=bool(result.detections),
        counts=result.counts_by_label,
        detections=[_to_detection_out(detection) for detection in result.detections],
        inference_ms=result.inference_ms,
    )
