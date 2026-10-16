"""Augmentation presets for training/finetune.py (Ultralytics train arguments).

Pick the one that matches where the camera is: augmentations should create
the variety your real images have, and nothing they never have.
"""

from __future__ import annotations

PRESETS: dict[str, dict[str, float]] = {
    # Ultralytics' own defaults.
    "default": {},
    # A fixed, level camera inside a house (e.g. a robot or a shelf camera):
    # - lighting changes a lot (day, lamps, dark rooms, IR night frames), so
    #   brightness and saturation vary strongly;
    # - the camera never rotates or turns upside down: no rotation, shear,
    #   perspective or vertical flip; left-right flip is fine;
    # - people and pets are near or far: scale and translate;
    # - mosaic helps small datasets, and is switched off for the last epochs
    #   so the model also sees whole, real frames.
    "indoor-camera": {
        "hsv_h": 0.01,
        "hsv_s": 0.5,
        "hsv_v": 0.6,
        "degrees": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "translate": 0.1,
        "scale": 0.4,
        "mosaic": 1.0,
        "close_mosaic": 10,
        "mixup": 0.0,
        "copy_paste": 0.0,
    },
    # Outdoor cameras: as above, plus a little rotation and colour shift
    # (weather, sun direction).
    "outdoor-camera": {
        "hsv_h": 0.02,
        "hsv_s": 0.6,
        "hsv_v": 0.5,
        "degrees": 3.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "translate": 0.1,
        "scale": 0.5,
        "mosaic": 1.0,
        "close_mosaic": 10,
    },
}


def preset(name: str) -> dict[str, float]:
    try:
        return dict(PRESETS[name])
    except KeyError:
        raise ValueError(f"unknown preset {name!r}; choose one of {', '.join(PRESETS)}") from None
