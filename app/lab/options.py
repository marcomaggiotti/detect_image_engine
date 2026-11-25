"""Describes every training option for the UI: default, range and meaning.

The web interfaces render their forms from ``GET /v1/lab/options``, so new
options only need to be added here (and handled in ``training/run_job.py``).
"""

from __future__ import annotations

from training.presets import PRESETS


def opt(key, default, *, kind="number", min=None, max=None, step=None, choices=None, help=""):
    return {
        "key": key,
        "default": default,
        "kind": kind,
        "min": min,
        "max": max,
        "step": step,
        "choices": choices,
        "help": help,
    }


BASE_MODELS = [
    {"weights": "yolo11n.pt", "task": "detect", "params_m": 2.6, "note": "fastest"},
    {"weights": "yolo11s.pt", "task": "detect", "params_m": 9.4, "note": ""},
    {"weights": "yolo11m.pt", "task": "detect", "params_m": 20.1, "note": "more accurate"},
    {"weights": "yolo11n-pose.pt", "task": "pose", "params_m": 2.9, "note": "17 body keypoints"},
    {"weights": "yolo11s-pose.pt", "task": "pose", "params_m": 9.9, "note": "17 body keypoints"},
    {"weights": "yolo11m-pose.pt", "task": "pose", "params_m": 20.9, "note": "17 body keypoints"},
    {"weights": "yolov8n.pt", "task": "detect", "params_m": 3.2, "note": "YOLOv8"},
    {
        "weights": "rtdetr-l.pt",
        "task": "detect",
        "params_m": 32.0,
        "arch": "rtdetr",
        "note": "transformer detector, Hungarian matching",
    },
]

GROUPS = [
    {
        "id": "transfer",
        "title": "Transfer learning & fine-tuning",
        "options": [
            opt(
                "pretrained",
                True,
                kind="bool",
                help="Start from COCO-pretrained weights (transfer learning). Off = random init.",
            ),
            opt(
                "freeze",
                0,
                min=0,
                max=24,
                step=1,
                help="Freeze the first N layers. 10 = the whole backbone: only the neck and head "
                "adapt, good with < 500 images.",
            ),
            opt("epochs", 50, min=1, max=1000, step=1),
            opt(
                "patience",
                20,
                min=0,
                max=300,
                step=1,
                help="Early stopping: epochs without validation improvement.",
            ),
            opt(
                "imgsz",
                640,
                min=160,
                max=1536,
                step=32,
                help="Training resolution. Larger helps small/distant dancers, costs time.",
            ),
            opt("batch", 16, min=-1, max=256, step=1, help="-1 = auto-fit GPU memory."),
            opt(
                "optimizer",
                "auto",
                kind="choice",
                choices=["auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"],
            ),
            opt("lr0", 0.01, min=0.00001, max=0.1, step=0.0001, help="Initial learning rate."),
            opt(
                "lrf",
                0.01,
                min=0.0001,
                max=1,
                step=0.001,
                help="Final learning rate as a fraction of lr0.",
            ),
            opt("cos_lr", False, kind="bool", help="Cosine learning-rate schedule."),
            opt("warmup_epochs", 3.0, min=0, max=10, step=0.5),
            opt("weight_decay", 0.0005, min=0, max=0.01, step=0.0001),
            opt("momentum", 0.937, min=0.5, max=0.999, step=0.001),
            opt("seed", 0, min=0, max=99999, step=1),
        ],
    },
    {
        "id": "loss",
        "title": "Losses",
        "options": [
            opt(
                "loss.box",
                7.5,
                min=0,
                max=20,
                step=0.1,
                help="Bounding-box regression gain (CIoU loss). Raise when boxes are loose.",
            ),
            opt(
                "loss.dfl",
                1.5,
                min=0,
                max=5,
                step=0.1,
                help="Distribution Focal Loss gain: sharpens box edge distributions.",
            ),
            opt(
                "loss.cls",
                0.5,
                min=0,
                max=5,
                step=0.05,
                help="Classification (and objectness, which YOLO11 folds into the class "
                "scores) gain. Raise when classes are confused.",
            ),
            opt(
                "loss.pose",
                12.0,
                min=0,
                max=30,
                step=0.5,
                help="Pose models: keypoint location (OKS) gain.",
            ),
            opt(
                "loss.kobj",
                1.0,
                min=0,
                max=5,
                step=0.1,
                help="Pose models: keypoint visibility gain.",
            ),
            opt(
                "loss.focal_gamma",
                0.0,
                min=0,
                max=5,
                step=0.1,
                help="Focal classification loss. 1.5-2 down-weights the many easy background "
                "anchors (class imbalance, foreground-background).",
            ),
            opt(
                "loss.focal_alpha",
                None,
                min=0,
                max=1,
                step=0.05,
                help="Optional balance between positives and negatives (e.g. 0.25).",
            ),
            opt(
                "loss.class_weights",
                None,
                kind="list",
                help="Per-class weight on the classification loss (see dataset stats).",
            ),
        ],
    },
    {
        "id": "assigner",
        "title": "Assignment strategy",
        "options": [
            opt(
                "assigner.topk",
                10,
                min=1,
                max=30,
                step=1,
                help="Task-Aligned Assigner: positive anchors per object. More = more "
                "supervision for small or crowded objects.",
            ),
            opt(
                "assigner.alpha",
                0.5,
                min=0,
                max=3,
                step=0.1,
                help="Weight of the classification score in the alignment metric.",
            ),
            opt(
                "assigner.beta",
                6.0,
                min=0,
                max=10,
                step=0.5,
                help="Weight of IoU in the alignment metric (localisation quality).",
            ),
            opt(
                "matcher.class",
                2.0,
                min=0,
                max=10,
                step=0.5,
                help="RT-DETR Hungarian matcher: class cost.",
            ),
            opt("matcher.bbox", 5.0, min=0, max=10, step=0.5, help="RT-DETR: L1 box cost."),
            opt("matcher.giou", 2.0, min=0, max=10, step=0.5, help="RT-DETR: GIoU cost."),
        ],
    },
    {
        "id": "augment",
        "title": "Augmentation",
        "options": [
            opt("augment.preset", "ballet-studio", kind="choice", choices=sorted(PRESETS)),
            opt(
                "augment.hsv_h", 0.015, min=0, max=0.1, step=0.005, help="Hue shift (stage lights)."
            ),
            opt("augment.hsv_s", 0.7, min=0, max=1, step=0.05, help="Saturation."),
            opt("augment.hsv_v", 0.4, min=0, max=1, step=0.05, help="Brightness."),
            opt("augment.degrees", 0.0, min=0, max=45, step=1, help="Rotation."),
            opt("augment.translate", 0.1, min=0, max=0.5, step=0.05),
            opt(
                "augment.scale",
                0.5,
                min=0,
                max=0.9,
                step=0.05,
                help="Zoom in/out (near/far dancers).",
            ),
            opt("augment.shear", 0.0, min=0, max=10, step=0.5),
            opt("augment.perspective", 0.0, min=0, max=0.001, step=0.0001),
            opt("augment.fliplr", 0.5, min=0, max=1, step=0.05, help="Left-right flip."),
            opt(
                "augment.flipud",
                0.0,
                min=0,
                max=1,
                step=0.05,
                help="Upside-down: keep 0 for dancers.",
            ),
            opt("augment.mosaic", 1.0, min=0, max=1, step=0.05, help="4-image mosaic."),
            opt(
                "augment.close_mosaic",
                10,
                min=0,
                max=50,
                step=1,
                help="Last N epochs without mosaic.",
            ),
            opt("augment.mixup", 0.0, min=0, max=1, step=0.05),
            opt("augment.copy_paste", 0.0, min=0, max=1, step=0.05, help="Segment datasets only."),
            opt(
                "augment.erasing",
                0.4,
                min=0,
                max=0.9,
                step=0.05,
                help="Random erasing (occlusion).",
            ),
        ],
    },
    {
        "id": "imbalance",
        "title": "Class imbalance & split",
        "options": [
            opt(
                "imbalance.rebalance",
                "none",
                kind="choice",
                choices=["none", "repeat-factor"],
                help="Repeat-factor sampling: images with rare classes appear more often.",
            ),
            opt(
                "imbalance.threshold",
                0.3,
                min=0.01,
                max=1,
                step=0.01,
                help="Classes in fewer than this share of images get repeated.",
            ),
            opt("imbalance.max_repeat", 4, min=1, max=10, step=1),
            opt("split.val_fraction", 0.2, min=0.05, max=0.5, step=0.05),
            opt(
                "split.strategy",
                "random",
                kind="choice",
                choices=["random", "by-day"],
                help="by-day keeps photos of one session together (no leakage).",
            ),
        ],
    },
    {
        "id": "eval",
        "title": "Validation & export",
        "options": [
            opt(
                "val.iou",
                0.7,
                min=0.1,
                max=0.95,
                step=0.05,
                help="NMS IoU threshold during validation.",
            ),
            opt(
                "val.conf",
                0.001,
                min=0.0001,
                max=0.5,
                step=0.001,
                help="Keep low for mAP; the best serving threshold is reported.",
            ),
            opt(
                "export",
                None,
                kind="choice",
                choices=[None, "onnx", "engine", "openvino", "torchscript"],
                help="Latency: export the best weights (engine = TensorRT, NVIDIA GPU).",
            ),
            opt("half", False, kind="bool", help="FP16 export (GPU)."),
        ],
    },
]


def options() -> dict:
    return {
        "base_models": BASE_MODELS,
        "groups": GROUPS,
        "presets": PRESETS,
        "tunable": sorted(tunable_options()),  # what a hyperparameter search may vary
    }


TUNABLE_KINDS = {"number", "bool", "choice"}
# Not searched: set by the search (epochs), or they would make the trials incomparable
# (another split, other validation thresholds) or do not change the training.
NOT_TUNABLE = {"epochs", "seed", "export", "half", "split.val_fraction", "split.strategy",
               "val.iou", "val.conf"}  # fmt: skip


def tunable_options() -> dict[str, dict]:
    """The options a hyperparameter search may vary, by key."""
    return {
        option["key"]: option
        for group in GROUPS
        for option in group["options"]
        if option["kind"] in TUNABLE_KINDS and option["key"] not in NOT_TUNABLE
    }


def nest(flat: dict) -> dict:
    """{"loss.box": 7.5} -> {"loss": {"box": 7.5}}"""
    out: dict = {}
    for key, value in flat.items():
        parts = key.split(".")
        target = out
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return out
