"""Run one lab job (train, validate or export) described by a JSON file.

Started by the lab API (``app/lab/jobs.py``) in a separate process:

    python -m training.run_job --job-dir data/jobs/<id>

Reads ``<job-dir>/config.json``, writes ``progress.json`` after every epoch
and ``summary.json`` at the end. It can also be run by hand.

Train config (all keys optional except ``dataset_dir``):

    {
      "kind": "train",
      "dataset_dir": "data/datasets/ballet",
      "base": "yolo11n.pt",         # transfer learning: pretrained weights or a previous best.pt
      "pretrained": true,           # false = train from scratch (uses the matching .yaml)
      "arch": "yolo",               # or "rtdetr"
      "epochs": 50, "imgsz": 640, "batch": 16, "device": null, "patience": 20,
      "freeze": 10,                 # fine-tuning: freeze the first N layers (10 = backbone)
      "optimizer": "auto", "lr0": 0.01, "lrf": 0.01, "momentum": 0.937,
      "weight_decay": 0.0005, "warmup_epochs": 3, "cos_lr": false, "seed": 0,
      "loss": {"box": 7.5, "cls": 0.5, "dfl": 1.5, "pose": 12.0, "kobj": 1.0,
               "focal_gamma": 0.0, "focal_alpha": null, "class_weights": null},
      "assigner": {"topk": 10, "alpha": 0.5, "beta": 6.0},
      "matcher": {"class": 2, "bbox": 5, "giou": 2},          # RT-DETR only
      "augment": {"preset": "ballet-studio", "fliplr": 0.5, ...},
      "split": {"val_fraction": 0.2, "strategy": "random", "seed": 0},
      "imbalance": {"rebalance": "repeat-factor", "threshold": 0.3, "max_repeat": 4},
      "val": {"conf": 0.001, "iou": 0.7},
      "export": null                # "onnx", "engine" (TensorRT), "openvino", ...
    }
"""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from pathlib import Path

from training.customize import LossCustomization, install
from training.finetune import best_confidence
from training.presets import preset
from training.splits import build_split

TRAIN_KEYS = (
    "epochs", "imgsz", "batch", "device", "patience", "freeze", "optimizer", "lr0", "lrf",
    "momentum", "weight_decay", "warmup_epochs", "cos_lr", "seed", "workers", "cache",
    "amp", "dropout", "rect", "single_cls", "multi_scale", "nbs", "plots", "val",
)  # fmt: skip
EXPORT_OPTION_KEYS = ("imgsz", "half", "int8", "dynamic", "simplify")
LOSS_GAINS = ("box", "cls", "dfl", "pose", "kobj")
AUGMENT_KEYS = (
    "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective",
    "flipud", "fliplr", "mosaic", "mixup", "copy_paste", "erasing", "close_mosaic",
    "auto_augment",
)  # fmt: skip


def _num(value):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else round(f, 6)


def write_json(path: Path, data) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    tmp.replace(path)


def train_arguments(config: dict) -> dict:
    """The keyword arguments passed to Ultralytics ``model.train``."""
    args = {key: config[key] for key in TRAIN_KEYS if config.get(key) is not None}
    loss = config.get("loss") or {}
    args.update({key: loss[key] for key in LOSS_GAINS if loss.get(key) is not None})
    augment = dict(config.get("augment") or {})
    args.update(preset(augment.pop("preset", None) or "default"))
    args.update({key: augment[key] for key in AUGMENT_KEYS if augment.get(key) is not None})
    return args


def loss_customization(config: dict) -> LossCustomization:
    loss = config.get("loss") or {}
    assigner = config.get("assigner") or {}
    matcher = config.get("matcher") or {}
    return LossCustomization(
        tal_topk=assigner.get("topk"),
        tal_alpha=assigner.get("alpha"),
        tal_beta=assigner.get("beta"),
        match_class=matcher.get("class"),
        match_bbox=matcher.get("bbox"),
        match_giou=matcher.get("giou"),
        focal_gamma=float(loss.get("focal_gamma") or 0.0),
        focal_alpha=loss.get("focal_alpha"),
        class_weights=loss.get("class_weights") or None,
    )


def base_weights(config: dict) -> str:
    base = config.get("base") or "yolo11n.pt"
    if config.get("pretrained", True) is False and base.endswith(".pt"):
        # From scratch: same architecture, random weights.
        return base[: -len(".pt")] + ".yaml"
    return base


def model_class(config: dict):
    from ultralytics import RTDETR, YOLO

    return RTDETR if (config.get("arch") or "yolo").lower() == "rtdetr" else YOLO


def box_metrics(metrics, names: dict) -> dict:
    """Overall and per-class metrics from an Ultralytics validation result."""
    result = {}
    for attr, part in (("box", "box"), ("pose", "pose")):
        m = getattr(metrics, attr, None)
        if m is None or not hasattr(m, "map"):
            continue
        result[part] = {
            "precision": _num(getattr(m, "mp", None)),
            "recall": _num(getattr(m, "mr", None)),
            "map50": _num(m.map50),
            "map50_95": _num(m.map),
        }
        try:
            per_class = {
                names[int(c)]: _num(v)
                for c, v in zip(m.ap_class_index, m.maps[m.ap_class_index], strict=False)
            }
        except Exception:  # noqa: BLE001 - optional detail
            per_class = {}
        result[part]["per_class_map50_95"] = per_class
    speed = getattr(metrics, "speed", None)
    if isinstance(speed, dict):
        result["speed_ms"] = {k: _num(v) for k, v in speed.items()}
    return result


def run_train(config: dict, job_dir: Path, log) -> dict:
    split_config = config.get("split") or {}
    imbalance = config.get("imbalance") or {}
    split = build_split(
        Path(config["dataset_dir"]),
        job_dir / "dataset",
        val_fraction=split_config.get("val_fraction", 0.2),
        strategy=split_config.get("strategy", "random"),
        seed=split_config.get("seed", 0),
        rebalance=imbalance.get("rebalance", "none"),
        rfs_threshold=imbalance.get("threshold", 0.3),
        max_repeat=imbalance.get("max_repeat", 4),
    )
    log(f"Dataset split: {json.dumps(split)}")
    progress = {"stage": "loading model", "split": split}
    write_json(job_dir / "progress.json", progress)

    cls = model_class(config)
    model = cls(base_weights(config))
    install(model, loss_customization(config), log=log)
    started = time.time()

    def on_fit_epoch_end(trainer):
        losses = {}
        try:
            losses = {
                k: _num(v)
                for k, v in trainer.label_loss_items(trainer.tloss, prefix="train").items()
            }
        except Exception:  # noqa: BLE001
            pass
        progress.update(
            stage="training",
            epoch=trainer.epoch + 1,
            epochs=trainer.epochs,
            elapsed_s=round(time.time() - started, 1),
            metrics={k: _num(v) for k, v in (trainer.metrics or {}).items()},
            losses=losses,
        )
        write_json(job_dir / "progress.json", progress)

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    args = train_arguments(config)
    log(f"Train arguments: {json.dumps(args)}")
    model.train(
        data=split["yaml"],
        project=str(job_dir.resolve()),
        name="train",
        exist_ok=True,
        **args,
    )
    best = Path(model.trainer.best)

    progress["stage"] = "validating"
    write_json(job_dir / "progress.json", progress)
    tuned = cls(best)
    val = config.get("val") or {}
    metrics = tuned.val(
        data=split["yaml"],
        imgsz=args.get("imgsz", 640),
        device=args.get("device"),
        conf=val.get("conf", 0.001),
        iou=val.get("iou", 0.7),
        project=str(job_dir.resolve()),
        name="val",
        exist_ok=True,
    )
    summary = {
        "weights": str(best.resolve()),
        "names": {int(k): v for k, v in tuned.names.items()},
        "task": tuned.task,
        "metrics": box_metrics(metrics, tuned.names),
        "best_confidence": best_confidence(metrics),
        "split": split,
        "train_seconds": round(time.time() - started, 1),
    }
    if config.get("export"):
        summary["exported"] = str(
            tuned.export(
                format=config["export"], imgsz=args.get("imgsz", 640), half=config.get("half")
            )
        )
    return summary


def run_export(config: dict, job_dir: Path, log) -> dict:
    model = model_class(config)(config["weights"])
    export_options = {key: config[key] for key in EXPORT_OPTION_KEYS if key in config}
    log(f"Exporting {config['weights']} to {config['format']} {export_options}")
    started = time.time()
    path = model.export(format=config["format"], **export_options)
    return {"exported": str(path), "seconds": round(time.time() - started, 1)}


def run_val(config: dict, job_dir: Path, log) -> dict:
    split = build_split(Path(config["dataset_dir"]), job_dir / "dataset", val_fraction=1.0)
    model = model_class(config)(config["weights"])
    metrics = model.val(
        data=split["yaml"],
        imgsz=config.get("imgsz", 640),
        conf=config.get("conf", 0.001),
        iou=config.get("iou", 0.7),
        project=str(job_dir.resolve()),
        name="val",
        exist_ok=True,
    )
    return {
        "metrics": box_metrics(metrics, model.names),
        "best_confidence": best_confidence(metrics),
    }


RUNNERS = {"train": run_train, "export": run_export, "val": run_val}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--job-dir", required=True, type=Path)
    args = p.parse_args(argv)
    job_dir: Path = args.job_dir
    config = json.loads((job_dir / "config.json").read_text())

    def log(message: str) -> None:
        print(message, flush=True)

    try:
        summary = RUNNERS[config.get("kind", "train")](config, job_dir, log)
    except Exception as exc:  # noqa: BLE001 - reported to the API
        traceback.print_exc()
        write_json(job_dir / "summary.json", {"error": f"{type(exc).__name__}: {exc}"})
        return 1
    write_json(job_dir / "summary.json", summary)
    log(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
