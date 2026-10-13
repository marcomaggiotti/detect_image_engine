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
RUN pip install ".[ultralytics,torchvision]"

# Bake the default pretrained model into the image. To serve another or a
# fine-tuned model, mount it (e.g. -v ./runs/pets-v1/weights:/models/custom)
# and set DETECT_MODEL_WEIGHTS.
RUN mkdir /models && cd /models && python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
ENV DETECT_MODEL_WEIGHTS=/models/yolo11n.pt

RUN useradd --system --no-create-home app
USER app

# Platforms such as Render inject the port to listen on through $PORT.
ENV PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ['PORT'] + '/health')"
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port \"$PORT\""]
