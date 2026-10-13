"""FastAPI application entry point: ``uvicorn app.main:app``."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __doc__ as description
from app.api.routes import router
from app.config import Settings, get_settings
from app.detectors import Detector, DetectorConfig, create_detector
from app.service import DetectionService

logger = logging.getLogger("detect_image_engine")


def build_detector(settings: Settings) -> Detector:
    config = DetectorConfig(
        weights=settings.model_weights,
        arch=settings.model_arch,
        labels_file=settings.model_labels,
        device=settings.device,
        iou_threshold=settings.iou_threshold,
        image_size=settings.image_size,
        max_detections=settings.max_detections,
    )
    return create_detector(settings.backend, config)


def create_app(settings: Settings | None = None, detector: Detector | None = None) -> FastAPI:
    """Build the app. ``detector`` can be injected (tests, embedding)."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        model = detector or build_detector(settings)
        app.state.settings = settings
        app.state.service = DetectionService(
            model,
            default_classes=settings.target_classes,
            confidence_threshold=settings.confidence_threshold,
            max_image_pixels=settings.max_image_pixels,
        )
        if settings.warmup:
            model.warmup()
        logger.info(
            "Loaded %s model %s (%d classes), default classes: %s",
            model.backend_name,
            settings.model_weights,
            len(model.labels),
            app.state.service.default_selection.names or "all",
        )
        yield

    app = FastAPI(
        title="Detect Image Engine",
        description=description,
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


def _default_app() -> FastAPI:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    return create_app()


app = _default_app()
