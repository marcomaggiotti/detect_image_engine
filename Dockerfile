FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_AUTOINSTALL=false \
    YOLO_CONFIG_DIR=/tmp/ultralytics

# OpenCV (pulled in by ultralytics) needs these shared libraries.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch keeps the image small. For GPU, build with
# --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124 (or similar).
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install torch torchvision --index-url ${TORCH_INDEX}

COPY pyproject.toml README.md ./
COPY app ./app
COPY training ./training
RUN pip install ".[ultralytics,torchvision]"

# Bake pretrained models into the image (space separated). To serve another or
# a fine-tuned model, mount it (e.g. -v ./runs/pets-v1/weights:/models/custom)
# and set DETECT_MODEL_WEIGHTS. yolo11n-pose.pt adds 17 body keypoints.
ARG BAKE_WEIGHTS="yolo11n.pt yolo11n-pose.pt"
RUN mkdir /models && cd /models && for w in $BAKE_WEIGHTS; do \
        python -c "from ultralytics import YOLO; YOLO('$w')"; done
ENV DETECT_MODEL_WEIGHTS=/models/yolo11n.pt

# The lab (datasets, training runs, downloaded/exported weights) writes here:
# mount a volume on /data to keep them.
RUN useradd --system --create-home app && mkdir -p /data && chown app /data
ENV DETECT_DATA_DIR=/data YOLO_CONFIG_DIR=/data/ultralytics
USER app

# Platforms such as Render inject the port to listen on through $PORT.
ENV PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ['PORT'] + '/health')"
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port \"$PORT\""]
