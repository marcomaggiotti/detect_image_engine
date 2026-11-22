"""Latency of the served model at several input sizes and precisions."""

from __future__ import annotations

import statistics
import time
from dataclasses import replace

from PIL import Image

from app.service import DetectionService

BENCHMARK_CONFIDENCE = 0.25


def measure_latency(
    service: DetectionService,
    image: Image.Image,
    *,
    image_sizes: list[int],
    half_precision_options: list[bool],
    runs: int,
) -> list[dict]:
    """One row per (precision, size): mean / p50 / p95 latency and frames per second.

    The detector's settings are restored afterwards. A setting the hardware does not
    support (e.g. FP16 on a CPU) gives a row with an ``error`` instead.
    """
    rows = []
    with service.exclusive_detector() as detector:
        original_config = detector.config
        try:
            for half in half_precision_options:
                for image_size in image_sizes:
                    detector.config = replace(original_config, image_size=image_size, half=half)
                    rows.append(_measure_one_setting(detector, image, image_size, half, runs))
        finally:
            detector.config = original_config
    return rows


def _measure_one_setting(
    detector, image: Image.Image, image_size: int, half: bool, runs: int
) -> dict:
    try:
        detector.predict(image, min_confidence=BENCHMARK_CONFIDENCE)  # warm-up
        durations_ms, detection_count = [], 0
        for _ in range(runs):
            started = time.perf_counter()
            detection_count = len(detector.predict(image, min_confidence=BENCHMARK_CONFIDENCE))
            durations_ms.append((time.perf_counter() - started) * 1000)
    except Exception as error:  # noqa: BLE001 - reported per row (e.g. FP16 on CPU)
        return {"image_size": image_size, "half": half, "error": str(error)}

    durations_ms.sort()
    mean_ms = statistics.fmean(durations_ms)
    p95_index = min(len(durations_ms) - 1, int(len(durations_ms) * 0.95))
    return {
        "image_size": image_size,
        "half": half,
        "mean_ms": round(mean_ms, 2),
        "p50_ms": round(durations_ms[len(durations_ms) // 2], 2),
        "p95_ms": round(durations_ms[p95_index], 2),
        "fps": round(1000 / mean_ms, 1),
        "detections": detection_count,
    }
