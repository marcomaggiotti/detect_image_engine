"""HTTP API of the lab, mounted under ``/v1/lab``."""

from fastapi import APIRouter

from app.lab.api import datasets, evaluation, jobs, runtime
from app.lab.api.errors import register_error_handlers

router = APIRouter(prefix="/v1/lab", tags=["lab"])
for module in (runtime, datasets, jobs, evaluation):
    router.include_router(module.router)

__all__ = ["register_error_handlers", "router"]
