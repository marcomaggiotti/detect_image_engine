"""Fine-tune an Ultralytics detector on your own data.

Example:
    python -m training.finetune --data training/dataset.example.yaml \
        --base yolo11n.pt --epochs 50 --name pets-v1

The best checkpoint is written under runs/<name>/weights/best.pt; the script
prints the environment variables to serve it. Requires the ultralytics extra.
"""

from __future__ import annotations

import argparse
from pathlib import Path


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
        project=str(Path(args.project).resolve()),
        name=args.name,
        exist_ok=True,
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
    return weights


if __name__ == "__main__":
    main()
