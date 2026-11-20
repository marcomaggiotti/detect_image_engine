"""Robustness check: how stable are detections under realistic image changes?

Each perturbation is applied at a severity from 1 (mild) to 5 (strong). The
detections on the perturbed image are compared with those on the clean image,
taken as the reference: a reference object is "kept" when a detection of the
same class overlaps it with IoU >= 0.5. Geometric perturbations (flip,
rotation) move the reference boxes with the image.
"""

from __future__ import annotations

import base64
import io
import math
import random
from collections.abc import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from app.detectors import Detection

Box = tuple[float, float, float, float]


def _brightness(im, s):
    return ImageEnhance.Brightness(im).enhance(1 - 0.15 * s), None


def _overexpose(im, s):
    return ImageEnhance.Brightness(im).enhance(1 + 0.25 * s), None


def _contrast(im, s):
    return ImageEnhance.Contrast(im).enhance(1 - 0.17 * s), None


def _blur(im, s):
    return im.filter(ImageFilter.GaussianBlur(radius=0.8 * s)), None


def _motion_blur(im, s):
    # Horizontal box blur of length 4*s+1: a dancer or camera moving sideways.
    length = 4 * s + 1
    arr = np.asarray(im).astype(np.float32)
    pad = np.pad(arr, ((0, 0), (length // 2, length // 2), (0, 0)), mode="edge")
    csum = np.cumsum(pad, axis=1)
    csum = np.concatenate([np.zeros_like(csum[:, :1]), csum], axis=1)
    out = (csum[:, length:] - csum[:, :-length]) / length
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), None


def _noise(im, s):
    arr = np.asarray(im).astype(np.float32)
    rng = np.random.default_rng(0)
    arr += rng.normal(0, 8 * s, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)), None


def _jpeg(im, s):
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=max(5, 60 - 11 * s))
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB"), None


def _low_resolution(im, s):
    w, h = im.size
    f = 1 / (1 + s)
    small = im.resize((max(8, int(w * f)), max(8, int(h * f))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR), None


def _occlusion(im, s):
    out = im.copy()
    draw = ImageDraw.Draw(out)
    w, h = im.size
    rng = random.Random(s)
    for _ in range(s * 2):
        bw, bh = rng.uniform(0.05, 0.15) * w, rng.uniform(0.05, 0.15) * h
        x, y = rng.uniform(0, w - bw), rng.uniform(0, h - bh)
        draw.rectangle((x, y, x + bw, y + bh), fill=(110, 110, 110))
    return out, None


def _hflip(im, s):
    w, _ = im.size

    def move(b: Box) -> Box:
        return (w - b[2], b[1], w - b[0], b[3])

    return im.transpose(Image.FLIP_LEFT_RIGHT), move


def _rotation(im, s):
    angle = 4.0 * s
    w, h = im.size
    cx, cy = w / 2, h / 2
    rad = math.radians(angle)
    cos, sin = math.cos(rad), math.sin(rad)

    def move(b: Box) -> Box:
        # PIL rotates counter-clockwise; y axis points down.
        pts = [(b[0], b[1]), (b[2], b[1]), (b[0], b[3]), (b[2], b[3])]
        xs, ys = [], []
        for x, y in pts:
            dx, dy = x - cx, y - cy
            xs.append(cx + dx * cos + dy * sin)
            ys.append(cy - dx * sin + dy * cos)
        return (max(0, min(xs)), max(0, min(ys)), min(w, max(xs)), min(h, max(ys)))

    return im.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0)), move


PERTURBATIONS: dict[str, Callable] = {
    "darken": _brightness,
    "overexpose": _overexpose,
    "low_contrast": _contrast,
    "gaussian_blur": _blur,
    "motion_blur": _motion_blur,
    "noise": _noise,
    "jpeg": _jpeg,
    "low_resolution": _low_resolution,
    "occlusion": _occlusion,
    "hflip": _hflip,
    "rotation": _rotation,
}


def iou(a: Box, b: Box) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def compare(reference: list[Detection], found: list[Detection], move=None, thr=0.5) -> dict:
    used: set[int] = set()
    ious, conf_deltas = [], []
    for ref in reference:
        box = move(ref.box) if move else ref.box
        best, best_iou = None, thr
        for i, d in enumerate(found):
            if i in used or d.label != ref.label:
                continue
            v = iou(box, d.box)
            if v >= best_iou:
                best, best_iou = i, v
        if best is not None:
            used.add(best)
            ious.append(best_iou)
            conf_deltas.append(found[best].confidence - ref.confidence)
    n = len(reference)
    return {
        "kept": round(len(ious) / n, 3) if n else None,
        "mean_iou": round(sum(ious) / len(ious), 3) if ious else None,
        "mean_conf_delta": round(sum(conf_deltas) / len(conf_deltas), 3) if conf_deltas else None,
        "new_detections": len(found) - len(used),
    }


def preview(image: Image.Image, size: int = 320) -> str:
    im = image.copy()
    im.thumbnail((size, size))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def evaluate(
    image: Image.Image,
    detect: Callable[[Image.Image], list[Detection]],
    perturbations: list[str],
    severities: list[int],
    with_previews: bool = True,
) -> dict:
    unknown = [p for p in perturbations if p not in PERTURBATIONS]
    if unknown:
        raise ValueError(f"unknown perturbation(s): {', '.join(unknown)}")
    reference = detect(image)
    rows = []
    for name in perturbations:
        for s in severities:
            perturbed, move = PERTURBATIONS[name](image, max(1, min(5, int(s))))
            found = detect(perturbed)
            rows.append(
                {
                    "perturbation": name,
                    "severity": s,
                    "detections": len(found),
                    **compare(reference, found, move),
                    "boxes": [
                        {"label": d.label, "confidence": round(d.confidence, 3), "box": d.box}
                        for d in found
                    ],
                    "preview": preview(perturbed) if with_previews else None,
                    "image_size": list(perturbed.size),
                }
            )
    kept = [r["kept"] for r in rows if r["kept"] is not None]
    return {
        "reference": [
            {"label": d.label, "confidence": round(d.confidence, 3), "box": d.box}
            for d in reference
        ],
        "results": rows,
        "robustness_score": round(sum(kept) / len(kept), 3) if kept else None,
    }
