"""Robustness check: how stable are the detections under realistic image changes?

Each perturbation is applied at a severity from 1 (mild) to 5 (strong). The detections
on the perturbed image are compared with those on the clean image, taken as the
reference: a reference object is *kept* when a detection of the same class overlaps
it with IoU >= 0.5. Geometric perturbations (flip, rotation) also return how they move
a box, so that the reference boxes are moved with the image before comparing.
"""

from __future__ import annotations

import base64
import io
import math
import random
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from app.detectors import Detection

Box = tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
BoxTransform = Callable[[Box], Box]
MIN_SEVERITY, MAX_SEVERITY = 1, 5
MATCH_IOU = 0.5


@dataclass(frozen=True)
class Perturbed:
    image: Image.Image
    # How the perturbation moves a box; None when boxes stay where they are.
    move_box: BoxTransform | None = None


# --- perturbations: (image, severity 1-5) -> Perturbed --------------------------------
def darken(image: Image.Image, severity: int) -> Perturbed:
    return Perturbed(ImageEnhance.Brightness(image).enhance(1 - 0.15 * severity))


def overexpose(image: Image.Image, severity: int) -> Perturbed:
    return Perturbed(ImageEnhance.Brightness(image).enhance(1 + 0.25 * severity))


def lower_contrast(image: Image.Image, severity: int) -> Perturbed:
    return Perturbed(ImageEnhance.Contrast(image).enhance(1 - 0.17 * severity))


def gaussian_blur(image: Image.Image, severity: int) -> Perturbed:
    return Perturbed(image.filter(ImageFilter.GaussianBlur(radius=0.8 * severity)))


def motion_blur(image: Image.Image, severity: int) -> Perturbed:
    """Horizontal box blur: a dancer or a camera moving sideways."""
    blur_length = 4 * severity + 1
    pixels = np.asarray(image).astype(np.float32)
    half = blur_length // 2
    padded = np.pad(pixels, ((0, 0), (half, half), (0, 0)), mode="edge")
    cumulative = np.cumsum(padded, axis=1)
    cumulative = np.concatenate([np.zeros_like(cumulative[:, :1]), cumulative], axis=1)
    blurred = (cumulative[:, blur_length:] - cumulative[:, :-blur_length]) / blur_length
    return Perturbed(Image.fromarray(np.clip(blurred, 0, 255).astype(np.uint8)))


def sensor_noise(image: Image.Image, severity: int) -> Perturbed:
    pixels = np.asarray(image).astype(np.float32)
    pixels += np.random.default_rng(seed=0).normal(0, 8 * severity, pixels.shape)
    return Perturbed(Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8)))


def jpeg_compression(image: Image.Image, severity: int) -> Perturbed:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=max(5, 60 - 11 * severity))
    return Perturbed(Image.open(io.BytesIO(buffer.getvalue())).convert("RGB"))


def low_resolution(image: Image.Image, severity: int) -> Perturbed:
    width, height = image.size
    scale = 1 / (1 + severity)
    small = image.resize((max(8, int(width * scale)), max(8, int(height * scale))), Image.BILINEAR)
    return Perturbed(small.resize((width, height), Image.BILINEAR))


def occlusion(image: Image.Image, severity: int) -> Perturbed:
    """Grey rectangles: other dancers, a barre or furniture in front of the body."""
    occluded = image.copy()
    draw = ImageDraw.Draw(occluded)
    width, height = image.size
    rng = random.Random(severity)
    for _ in range(2 * severity):
        block_width, block_height = (
            rng.uniform(0.05, 0.15) * width,
            rng.uniform(0.05, 0.15) * height,
        )
        left, top = rng.uniform(0, width - block_width), rng.uniform(0, height - block_height)
        draw.rectangle((left, top, left + block_width, top + block_height), fill=(110, 110, 110))
    return Perturbed(occluded)


def horizontal_flip(image: Image.Image, severity: int) -> Perturbed:
    width = image.width

    def flip_box(box: Box) -> Box:
        x1, y1, x2, y2 = box
        return (width - x2, y1, width - x1, y2)

    return Perturbed(image.transpose(Image.FLIP_LEFT_RIGHT), flip_box)


def rotation(image: Image.Image, severity: int) -> Perturbed:
    """Counter-clockwise rotation by 4 degrees per severity level (a tilted phone)."""
    angle = math.radians(4.0 * severity)
    width, height = image.size
    center_x, center_y = width / 2, height / 2
    cos_angle, sin_angle = math.cos(angle), math.sin(angle)

    def rotate_box(box: Box) -> Box:
        # The bounding box of the rotated corners (the y axis points down).
        x1, y1, x2, y2 = box
        rotated_x, rotated_y = [], []
        for x, y in ((x1, y1), (x2, y1), (x1, y2), (x2, y2)):
            dx, dy = x - center_x, y - center_y
            rotated_x.append(center_x + dx * cos_angle + dy * sin_angle)
            rotated_y.append(center_y - dx * sin_angle + dy * cos_angle)
        return (
            max(0, min(rotated_x)),
            max(0, min(rotated_y)),
            min(width, max(rotated_x)),
            min(height, max(rotated_y)),
        )

    rotated = image.rotate(math.degrees(angle), resample=Image.BILINEAR, fillcolor=(0, 0, 0))
    return Perturbed(rotated, rotate_box)


# Names used by the API and the web interface.
PERTURBATIONS: dict[str, Callable[[Image.Image, int], Perturbed]] = {
    "darken": darken,
    "overexpose": overexpose,
    "low_contrast": lower_contrast,
    "gaussian_blur": gaussian_blur,
    "motion_blur": motion_blur,
    "noise": sensor_noise,
    "jpeg": jpeg_compression,
    "low_resolution": low_resolution,
    "occlusion": occlusion,
    "hflip": horizontal_flip,
    "rotation": rotation,
}


# --- comparison ---------------------------------------------------------------------
def iou(a: Box, b: Box) -> float:
    """Intersection over union of two boxes (0 = disjoint, 1 = identical)."""
    overlap_width = min(a[2], b[2]) - max(a[0], b[0])
    overlap_height = min(a[3], b[3]) - max(a[1], b[1])
    if overlap_width <= 0 or overlap_height <= 0:
        return 0.0
    intersection = overlap_width * overlap_height
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def compare_with_reference(
    reference: list[Detection],
    found: list[Detection],
    move_box: BoxTransform | None = None,
    min_iou: float = MATCH_IOU,
) -> dict:
    """Greedy one-to-one matching of reference objects with the new detections."""
    matched_indexes: set[int] = set()
    matched_ious, confidence_changes = [], []
    for expected in reference:
        expected_box = move_box(expected.box) if move_box else expected.box
        best_index, best_iou = None, min_iou
        for index, candidate in enumerate(found):
            if index in matched_indexes or candidate.label != expected.label:
                continue
            overlap = iou(expected_box, candidate.box)
            if overlap >= best_iou:
                best_index, best_iou = index, overlap
        if best_index is not None:
            matched_indexes.add(best_index)
            matched_ious.append(best_iou)
            confidence_changes.append(found[best_index].confidence - expected.confidence)
    return {
        "kept": _rounded_ratio(len(matched_ious), len(reference)),
        "mean_iou": _rounded_mean(matched_ious),
        "mean_conf_delta": _rounded_mean(confidence_changes),
        "new_detections": len(found) - len(matched_indexes),
    }


def _rounded_ratio(part: int, total: int) -> float | None:
    return round(part / total, 3) if total else None


def _rounded_mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def preview_data_url(image: Image.Image, max_side: int = 320) -> str:
    """A small JPEG of the image as a data URL, for the web interface."""
    thumbnail = image.copy()
    thumbnail.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    thumbnail.save(buffer, format="JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def _describe(detections: list[Detection]) -> list[dict]:
    return [
        {
            "label": detection.label,
            "confidence": round(detection.confidence, 3),
            "box": detection.box,
        }
        for detection in detections
    ]


def evaluate(
    image: Image.Image,
    detect: Callable[[Image.Image], list[Detection]],
    perturbation_names: list[str],
    severities: list[int],
    with_previews: bool = True,
) -> dict:
    """Run ``detect`` on the clean image and on every (perturbation, severity) pair."""
    unknown_names = [name for name in perturbation_names if name not in PERTURBATIONS]
    if unknown_names:
        raise ValueError(f"unknown perturbation(s): {', '.join(unknown_names)}")
    reference = detect(image)
    rows = []
    for name in perturbation_names:
        for severity in severities:
            clamped_severity = max(MIN_SEVERITY, min(MAX_SEVERITY, int(severity)))
            perturbed = PERTURBATIONS[name](image, clamped_severity)
            found = detect(perturbed.image)
            rows.append(
                {
                    "perturbation": name,
                    "severity": severity,
                    "detections": len(found),
                    **compare_with_reference(reference, found, perturbed.move_box),
                    "boxes": _describe(found),
                    "preview": preview_data_url(perturbed.image) if with_previews else None,
                    "image_size": list(perturbed.image.size),
                }
            )
    kept_ratios = [row["kept"] for row in rows if row["kept"] is not None]
    return {
        "reference": _describe(reference),
        "results": rows,
        "robustness_score": _rounded_mean(kept_ratios),
    }
