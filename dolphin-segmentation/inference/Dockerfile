# ===========================================================================
# Multi-stage Dockerfile — Dolphin Fin Segmentation Inference Service
# ===========================================================================
#
# Stage 1 (builder): install Python dependencies into a venv
# Stage 2 (runtime): copy venv + app, run as non-root
#
# ⚠️  PLACEHOLDER MODEL NOTE
# The container pulls the pretrained YOLOv8n-seg weights at first run
# (auto-downloaded by ultralytics to /weights/).
#
# To use a custom-trained checkpoint after Assignment 2/3:
#   1. Place your .pt file in ./weights/ on the host.
#   2. Uncomment the COPY line below or mount it via docker-compose volume:
#        volumes:
#          - ./weights:/weights
#   3. Set the env var:  MODEL_PATH=/weights/dolphin_fin_yolo_seg_v1.pt
# ===========================================================================

# ─── Stage 1: builder ──────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System libs required by OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir -r requirements.txt


# ─── Stage 2: runtime ──────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Minimal runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Create non-root user
RUN useradd --create-home appuser
USER appuser

# App source
COPY --chown=appuser:appuser app/ ./app/

# Ultralytics model cache directory (auto-download on first call)
ENV YOLO_CONFIG_DIR=/tmp/ultralytics
# ⚠️ PLACEHOLDER: set MODEL_PATH here or via docker-compose env to use custom weights
ENV MODEL_NAME=yolov8n-seg.pt
ENV MODEL_PATH=yolov8n-seg.pt
ENV MODEL_DEVICE=cpu

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
