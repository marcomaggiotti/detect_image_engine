"""Map the lab's domain errors to HTTP status codes, once for every route."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.imaging import InvalidImageError
from app.lab.datasets import DatasetError, DatasetNotFoundError
from app.lab.jobs import JobConflictError, JobNotFoundError
from app.lab.state import WeightsNotFoundError
from app.service import UnknownClassError

STATUS_BY_ERROR: dict[type[Exception], int] = {
    DatasetNotFoundError: 404,
    JobNotFoundError: 404,
    JobConflictError: 409,
    DatasetError: 400,
    WeightsNotFoundError: 400,
    UnknownClassError: 400,
    InvalidImageError: 400,
}


def register_error_handlers(app: FastAPI) -> None:
    for error_type, status_code in STATUS_BY_ERROR.items():
        app.add_exception_handler(error_type, _handler(status_code))


def _handler(status_code: int):
    async def handle(_request: Request, error: Exception) -> JSONResponse:
        return JSONResponse(status_code=status_code, content={"detail": str(error)})

    return handle
