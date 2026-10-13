# detect_image_engine

A microservice that receives an image over HTTP and returns the objects detected in it.

- **Stage 1:** finds **cats** by default.
- **Stage 2:** detects people, other pets or other objects. This needs a config change, a request parameter, or a fine-tuned model. No code changes.
- **Model independent:** the backend (YOLO, RT-DETR, Faster R-CNN, your own) and its weights come from configuration. Fine-tuned weights are a drop-in replacement.

```
           POST /v1/detect (image)
                    │
             ┌──────▼──────┐   validates classes, size limits,
             │   FastAPI   │   EXIF orientation, RGB conversion
             └──────┬──────┘
             ┌──────▼──────┐
             │  Detection  │   class selection, thresholds,
             │   Service   │   counts, timing, thread safety
             └──────┬──────┘
                    │  Detector interface (app/detectors/base.py)
       ┌────────────┼──────────────────────┐
┌──────▼──────┐ ┌───▼─────────┐ ┌──────────▼───────────┐
│ ultralytics │ │ torchvision │ │ your backend         │
│ YOLO/RT-DETR│ │ Faster R-CNN│ │ "pkg.module:Class"   │
└─────────────┘ └─────────────┘ └──────────────────────┘
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[ultralytics,dev]"
uvicorn app.main:app --reload
```

The first start downloads `yolo11n.pt` (a COCO-pretrained YOLO11 nano, ~5 MB). Interactive docs: http://localhost:8000/docs

```bash
curl -F image=@my_cat.jpg http://localhost:8000/v1/detect
```

```json
{
  "model": {"backend": "ultralytics", "weights": "yolo11n.pt", "version": null},
  "image": {"width": 335, "height": 500},
  "classes": ["cat"],
  "found": true,
  "counts": {"cat": 1},
  "detections": [
    {"label": "cat", "class_id": 15, "confidence": 0.91,
     "box": {"x1": 12.4, "y1": 30.1, "x2": 320.8, "y2": 488.0}}
  ],
  "inference_ms": 41.2
}
```

Box coordinates are pixels of the uploaded image, after EXIF rotation is applied.

## API

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/v1/detect` | Multipart upload, field `image`. Returns detections. |
| `GET` | `/v1/model` | Active backend, weights, version, default classes and every label the model knows. |
| `GET` | `/health` | Liveness check. |

`/v1/detect` query parameters:

| Param | Example | Meaning |
| --- | --- | --- |
| `classes` | `cat` · `cat,dog,person` · `*` | Classes to report. Defaults to `DETECT_TARGET_CLASSES`. `*` returns everything the model knows. Case-insensitive. An unknown class returns **400**. |
| `confidence` | `0.5` | Minimum score (0–1). Defaults to `DETECT_CONFIDENCE_THRESHOLD`. |

Errors: `400` for an invalid/corrupt image or unknown class, `413` for an upload over the size limit, `422` for an invalid parameter.

## Configuration

Everything is set with environment variables (or a `.env` file). See [`.env.example`](.env.example).

| Variable | Default | Description |
| --- | --- | --- |
| `DETECT_BACKEND` | `ultralytics` | `ultralytics`, `torchvision`, or a custom `package.module:ClassName` |
| `DETECT_MODEL_WEIGHTS` | `yolo11n.pt` | Model name or path to (fine-tuned) weights |
| `DETECT_MODEL_ARCH` | – | ultralytics: `yolo`/`rtdetr`; torchvision: e.g. `fasterrcnn_resnet50_fpn_v2` |
| `DETECT_MODEL_LABELS` | – | Label file (one name per line, line N = class N) |
| `DETECT_MODEL_VERSION` | – | Tag echoed in responses, so you know which model answered |
| `DETECT_DEVICE` | auto | `cpu`, `cuda`, `cuda:0`, `mps` |
| `DETECT_TARGET_CLASSES` | `cat` | Default classes, comma separated (`*` = all) |
| `DETECT_CONFIDENCE_THRESHOLD` | `0.25` | Default minimum score |
| `DETECT_IOU_THRESHOLD` | `0.45` | NMS IoU (ultralytics) |
| `DETECT_IMAGE_SIZE` | `640` | Inference size (ultralytics) |
| `DETECT_MAX_DETECTIONS` | `100` | Max boxes per image |
| `DETECT_MAX_UPLOAD_BYTES` | `10485760` | Upload size limit |
| `DETECT_MAX_IMAGE_PIXELS` | `40000000` | Rejects decompression bombs before decoding |
| `DETECT_WARMUP` | `true` | Run one dummy inference at startup |

At startup the service checks that the model can detect every class in `DETECT_TARGET_CLASSES`. If it can't, startup fails with a clear error instead of returning empty results.

## Going from cats to other objects

**1. Classes the model already knows: no training needed.** The default model is trained on COCO's 80 classes. That includes `person`, `cat`, `dog`, `bird`, `horse`, `sheep`, `cow`, `teddy bear`, `backpack`, `chair`, `cell phone` and more (see `GET /v1/model`). Pick classes per request:

```bash
curl -F image=@photo.jpg "http://localhost:8000/v1/detect?classes=cat,dog,person"
```

or change the service default: `DETECT_TARGET_CLASSES=cat,dog,person`.

**2. Better accuracy: swap the model.** Use a larger pretrained model, e.g. `DETECT_MODEL_WEIGHTS=yolo11m.pt`. Or switch family: `DETECT_MODEL_WEIGHTS=rtdetr-l.pt DETECT_MODEL_ARCH=rtdetr`. Or use torchvision (see below).

**3. New classes or your own data: fine-tune.** Examples: a specific cat ("luna"), a breed, rabbits, your products. See next section.

## Fine-tuning

1. Label images in YOLO format. CVAT, Label Studio and Roboflow all export it. See [`training/dataset.example.yaml`](training/dataset.example.yaml) for the layout.
2. Train, starting from a pretrained model (transfer learning):

   ```bash
   python -m training.finetune --data path/to/dataset.yaml \
       --base yolo11n.pt --epochs 50 --name pets-v1
   # small dataset? freeze the backbone:  --freeze 10
   # continue from a previous run:        --base runs/pets-v1/weights/best.pt
   # export for faster CPU/GPU serving:   --export onnx
   ```

   The script trains, validates (prints mAP), and prints the exact variables to serve the result.
3. Serve it:

   ```bash
   DETECT_MODEL_WEIGHTS=runs/pets-v1/weights/best.pt \
   DETECT_MODEL_VERSION=pets-v1 \
   DETECT_TARGET_CLASSES=cat,dog,person \
   uvicorn app.main:app
   ```

Fine-tuned Ultralytics weights store their own class names, so no label file is needed.

> ⚠️ Fine-tuning **replaces** the model's class list with your dataset's classes. If you still need `person` after training on pets, include `person` with labelled examples in the dataset.

## Changing the model backend

### torchvision (Faster R-CNN, RetinaNet, FCOS, SSD)

```bash
pip install -e ".[torchvision]"

# COCO-pretrained
DETECT_BACKEND=torchvision DETECT_MODEL_ARCH=fasterrcnn_resnet50_fpn_v2 \
DETECT_MODEL_WEIGHTS=DEFAULT uvicorn app.main:app

# fine-tuned with your own torchvision training loop
DETECT_BACKEND=torchvision DETECT_MODEL_ARCH=fasterrcnn_resnet50_fpn_v2 \
DETECT_MODEL_WEIGHTS=models/pets.pt DETECT_MODEL_LABELS=models/pets_labels.txt \
uvicorn app.main:app
```

A fine-tuned checkpoint is a `state_dict` (or `{"model": state_dict}`). The label file starts with `__background__` on line 0, then one class per line, matching the `num_classes` you trained with.

### Your own backend

Subclass `Detector` and point the config at it. No changes to the service are needed:

```python
# mypkg/detector.py
from app.detectors import Detection, Detector

class MyDetector(Detector):
    backend_name = "my-model"

    def __init__(self, config):
        super().__init__(config)
        self.model = load_my_model(config.weights)

    @property
    def labels(self) -> dict[int, str]:
        return {0: "cat", 1: "dog"}

    def detect(self, image, *, confidence, class_ids=None):
        ...  # return list[Detection] in pixel xyxy
```

```bash
DETECT_BACKEND=mypkg.detector:MyDetector uvicorn app.main:app
```

Open-vocabulary detectors (OWL-ViT, Grounding DINO, YOLO-World) also fit this interface. Use one if you want to search for arbitrary text labels without training.

## Docker

```bash
docker build -t detect-image-engine .
docker run -p 8000:8000 detect-image-engine

# serve a fine-tuned model
docker run -p 8000:8000 -v "$PWD/runs/pets-v1/weights:/models/custom:ro" \
  -e DETECT_MODEL_WEIGHTS=/models/custom/best.pt -e DETECT_MODEL_VERSION=pets-v1 \
  -e DETECT_TARGET_CLASSES=cat,dog,person detect-image-engine
```

The image uses CPU-only PyTorch. For GPU, pass `--build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124` and run with `--gpus all -e DETECT_DEVICE=cuda`.

Each process loads one model and runs one inference at a time. Scale with more replicas (or `uvicorn --workers N`; each worker loads its own copy of the model).

## Development

```bash
pip install -e ".[ultralytics,torchvision,dev]"
pytest
ruff check . && ruff format --check .
```

The API tests use a fake detector, so they need no ML framework. The backend smoke tests build randomly initialised models offline, and are skipped when the framework isn't installed.

## Licensing note

This repository is MIT-licensed. The **Ultralytics** package and its YOLO weights are **AGPL-3.0**. Running them in a network service may require releasing your service's source code under the AGPL, unless you hold an Ultralytics Enterprise license. If that is a problem, use the `torchvision` backend (BSD-3) or another permissively licensed model through a custom backend.
