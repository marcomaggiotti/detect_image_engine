from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.config import Settings
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


def _parse_classes(raw: list[str] | None) -> list[str] | None:
    """Accept both ?classes=cat,dog and ?classes=cat&classes=dog."""
    if not raw:
        return None
    names = [part.strip() for item in raw for part in item.split(",") if part.strip()]
    return names or None


def _box(xyxy: tuple[float, float, float, float]) -> Box:
    x1, y1, x2, y2 = (round(v, 1) for v in xyxy)
    return Box(x1=x1, y1=y1, x2=x2, y2=y2)


@router.get("/health", tags=["ops"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/v1/model", response_model=ModelInfo, tags=["detection"])
def model_info(request: Request) -> ModelInfo:
    service = _service(request)
    return ModelInfo(
        **_model_ref(request).model_dump(),
        arch=_settings(request).model_arch,
        default_classes=service.default_selection.names,
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

    data = await image.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(413, f"Image exceeds {max_bytes} bytes")

    try:
        result = await run_in_threadpool(
            service.detect, data, classes=_parse_classes(classes), confidence=confidence
        )
    except UnknownClassError as exc:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, f"{exc}. See GET /v1/model for available labels."
        ) from exc
    except InvalidImageError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    return DetectResponse(
        model=_model_ref(request),
        image=ImageInfo(width=result.width, height=result.height),
        classes=result.classes,
        found=bool(result.detections),
        counts=result.counts,
        detections=[
            DetectionOut(
                label=d.label,
                class_id=d.class_id,
                confidence=round(d.confidence, 4),
                box=_box(d.box),
            )
            for d in result.detections
        ],
        inference_ms=result.inference_ms,
    )
