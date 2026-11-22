"""Datasets: images, annotation, statistics, pre-labelling, zip import and export."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, Response, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from app.imaging import InvalidImageError
from app.lab.api.dependencies import CurrentSettings, Lab, Service, read_upload
from app.lab.api.requests import DatasetClasses, DatasetCreate, ImageLabels, PrelabelRequest
from app.lab.datasets import DatasetError, thumbnail
from app.lab.prelabel import classes_known_by_model, prelabel_dataset

router = APIRouter(prefix="/datasets")


@router.get("")
def list_datasets(lab: Lab) -> list[dict]:
    return lab.datasets.list()


@router.post("", status_code=201)
def create_dataset(request: DatasetCreate, lab: Lab) -> dict:
    return lab.datasets.create(request.name, request.classes, request.task, request.description)


@router.get("/{dataset_id}")
def get_dataset(dataset_id: str, lab: Lab) -> dict:
    return {**lab.datasets.metadata(dataset_id), "images": lab.datasets.images(dataset_id)}


@router.delete("/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: str, lab: Lab) -> Response:
    lab.datasets.delete(dataset_id)
    return Response(status_code=204)


@router.put("/{dataset_id}/classes")
def update_classes(dataset_id: str, request: DatasetClasses, lab: Lab) -> dict:
    return lab.datasets.update_classes(dataset_id, request.classes)


@router.get("/{dataset_id}/stats")
def dataset_stats(
    dataset_id: str, lab: Lab, power: Annotated[float, Query(ge=0, le=2)] = 0.5
) -> dict:
    """Class balance and suggested class weights (``power`` 1 = inverse frequency)."""
    return lab.datasets.stats(dataset_id, power)


@router.post("/{dataset_id}/images", status_code=201)
async def upload_images(
    dataset_id: str,
    files: Annotated[list[UploadFile], File()],
    lab: Lab,
    settings: CurrentSettings,
) -> dict:
    """Add pictures; invalid ones are reported per file instead of failing the upload."""
    lab.datasets.metadata(dataset_id)  # 404 before reading any file
    saved, errors = [], []
    for upload in files:
        try:
            content = await read_upload(upload, settings)
            saved.append(
                lab.datasets.add_image(dataset_id, upload.filename or "image.jpg", content)
            )
        except (HTTPException, DatasetError, InvalidImageError) as error:
            message = error.detail if isinstance(error, HTTPException) else str(error)
            errors.append({"file": upload.filename, "error": message})
    return {"saved": saved, "errors": errors}


@router.get("/{dataset_id}/images/{filename}")
def get_image(dataset_id: str, filename: str, lab: Lab, thumb: bool = False):
    path = lab.datasets.image_file(dataset_id, filename)
    if thumb:
        return Response(thumbnail(path), media_type="image/jpeg")
    return FileResponse(path)


@router.delete("/{dataset_id}/images/{filename}", status_code=204)
def delete_image(dataset_id: str, filename: str, lab: Lab) -> Response:
    lab.datasets.delete_image(dataset_id, filename)
    return Response(status_code=204)


@router.get("/{dataset_id}/labels/{filename}")
def get_labels(dataset_id: str, filename: str, lab: Lab) -> dict:
    boxes = lab.datasets.labels(dataset_id, filename)
    return {"annotated": boxes is not None, "boxes": boxes or []}


@router.put("/{dataset_id}/labels/{filename}")
def save_labels(dataset_id: str, filename: str, labels: ImageLabels, lab: Lab) -> dict:
    boxes = [box.model_dump(exclude_none=True) for box in labels.boxes]
    return {"saved": lab.datasets.save_labels(dataset_id, filename, boxes)}


@router.delete("/{dataset_id}/labels/{filename}", status_code=204)
def clear_labels(dataset_id: str, filename: str, lab: Lab) -> Response:
    lab.datasets.clear_labels(dataset_id, filename)
    return Response(status_code=204)


@router.post("/{dataset_id}/autolabel")
async def prelabel(
    dataset_id: str, lab: Lab, service: Service, request: PrelabelRequest | None = None
) -> dict:
    """Pre-label pictures with the active model (dataset classes matched by name)."""
    request = request or PrelabelRequest()
    dataset_classes = lab.datasets.metadata(dataset_id)["classes"]
    if not classes_known_by_model(dataset_classes, service):
        raise HTTPException(400, "the active model knows none of this dataset's classes")
    return await run_in_threadpool(
        prelabel_dataset,
        lab.datasets,
        service,
        dataset_id,
        min_confidence=request.confidence,
        overwrite=request.overwrite,
        only_files=request.files,
    )


@router.post("/{dataset_id}/import")
async def import_zip(dataset_id: str, archive: Annotated[UploadFile, File()], lab: Lab) -> dict:
    return lab.datasets.import_zip(dataset_id, await archive.read())


@router.get("/{dataset_id}/export")
def export_zip(dataset_id: str, lab: Lab) -> Response:
    return Response(
        lab.datasets.export_zip(dataset_id),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{dataset_id}.zip"'},
    )
