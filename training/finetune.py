"""Fine-tune an Ultralytics detector on your own data.

Example:
    python -m training.finetune --data training/dataset.example.yaml \
        --base yolo11n.pt --epochs 50 --name pets-v1 --preset indoor-camera

The best checkpoint is written under runs/<name>/weights/best.pt; the script
prints the environment variables to serve it. Requires the ultralytics extra.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from training.presets import PRESETS, preset


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data", required=True, help="Dataset YAML in Ultralytics/YOLO format")
    p.add_argument(
        "--base",
        default="yolo11n.pt",
        help="Starting weights: a pretrained model (yolo11s.pt, rtdetr-l.pt...) or a previous "
        "best.pt to keep improving it",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None, help="cpu, 0, 0,1, mps... (auto by default)")
    p.add_argument(
        "--freeze",
        type=int,
        default=None,
        help="Freeze the first N layers (e.g. 10 = backbone) for small datasets",
    )
    p.add_argument(
        "--preset",
        default="default",
        choices=sorted(PRESETS),
        help="Augmentations matching the camera (training/presets.py), e.g. indoor-camera",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Stop when validation has not improved for this many epochs (0 = never)",
    )
    p.add_argument("--project", default="runs", help="Output directory")
    p.add_argument("--name", default="finetune", help="Run name, used as model version")
    p.add_argument("--rtdetr", action="store_true", help="Base weights are an RT-DETR model")
    p.add_argument(
        "--export",
        default=None,
        help="Also export the best model, e.g. onnx, openvino, engine",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    from ultralytics import RTDETR, YOLO

    model = (RTDETR if args.rtdetr else YOLO)(args.base)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        freeze=args.freeze,
        patience=args.patience,
        project=str(Path(args.project).resolve()),
        name=args.name,
        exist_ok=True,
        **preset(args.preset),
    )
    best = Path(model.trainer.best)

    tuned = (RTDETR if args.rtdetr else YOLO)(best)
    metrics = tuned.val(
        data=args.data,
        imgsz=args.imgsz,
        device=args.device,
        project=str(Path(args.project).resolve()),
        name=f"{args.name}-val",
        exist_ok=True,
    )
    print(f"\nValidation mAP50-95: {metrics.box.map:.3f}  mAP50: {metrics.box.map50:.3f}")
    print(f"Classes: {tuned.names}")
    threshold = best_confidence(metrics)
    if threshold is not None:
        print(f"Best confidence threshold on the validation set (highest F1): {threshold:.2f}")

    weights = best
    if args.export:
        weights = Path(tuned.export(format=args.export, imgsz=args.imgsz))

    print("\nServe it with:")
    print("  DETECT_BACKEND=ultralytics")
    print(f"  DETECT_MODEL_WEIGHTS={weights}")
    if args.rtdetr:
        print("  DETECT_MODEL_ARCH=rtdetr")
    print(f"  DETECT_MODEL_VERSION={args.name}")
    print(f"  DETECT_TARGET_CLASSES={','.join(tuned.names.values())}")
    if threshold is not None:
        print(f"  DETECT_CONFIDENCE_THRESHOLD={threshold:.2f}")
    return weights


def best_confidence(metrics) -> float | None:
    """The confidence where F1 (precision and recall together) peaks on the
    validation set, averaged over classes: a good default threshold."""
    try:
        f1 = [list(row) for row in metrics.box.f1_curve]
        px = list(metrics.box.px)
    except (AttributeError, TypeError):
        return None
    if not f1 or not px:
        return None
    mean = [sum(column) / len(f1) for column in zip(*f1, strict=False)]
    return float(px[max(range(len(mean)), key=mean.__getitem__)])


if __name__ == "__main__":
    main()
