"""Shared objects of the lab routes, injected with FastAPI's ``Depends``."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request, UploadFile
from PIL import Image

from app.config import Settings
from app.imaging import decode_image
from app.lab.state import LabState
from app.service import DetectionService


def get_lab(request: Request) -> LabState:
    return request.app.state.lab


def get_service(request: Request) -> DetectionService:
    return request.app.state.service


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


Lab = Annotated[LabState, Depends(get_lab)]
Service = Annotated[DetectionService, Depends(get_service)]
CurrentSettings = Annotated[Settings, Depends(get_settings)]


async def read_upload(upload: UploadFile, settings: Settings) -> bytes:
    """The uploaded bytes, refusing files over the configured size limit."""
    content = await upload.read(settings.max_upload_bytes + 1)
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            413, f"{upload.filename or 'file'} is larger than {settings.max_upload_bytes} bytes"
        )
    return content


async def read_uploaded_image(upload: UploadFile, settings: Settings) -> Image.Image:
    return decode_image(await read_upload(upload, settings), max_pixels=settings.max_image_pixels)


def split_csv(value: str | None) -> list[str]:
    """``"a, b,,c"`` -> ``["a", "b", "c"]``."""
    return [part.strip() for part in (value or "").split(",") if part.strip()]
