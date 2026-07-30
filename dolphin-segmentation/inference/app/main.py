"""
FastAPI application for Dolphin Fin Instance Segmentation.

Endpoints:
  POST /predict  — run inference on an uploaded image
  GET  /health   — liveness / model status check

# ⚠️ PLACEHOLDER MODEL NOTE
# The current deployment uses a pretrained YOLOv8n-seg (COCO) model.
# All API contracts, schemas, and data flows are production-ready; only
# the model weights need replacing with the custom-trained dolphin-fin
# checkpoint after Assignment 2/3.
"""

import base64
import io
import uuid
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .inference import get_model, is_model_loaded, run_inference, MODEL_NAME
from .schemas import (
    FinInstance,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: pre-warm model (modern FastAPI pattern, replaces deprecated
# @app.on_event("startup"))
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Pre-warming model on startup …")
    try:
        get_model()
        logger.info("Model ready.")
    except Exception as exc:
        # Don't crash the server — /health will report model_loaded=False
        logger.error("Model failed to load on startup: %s", exc)
    yield
    # Shutdown: nothing to clean up for YOLO


app = FastAPI(
    title="Dolphin Fin Instance Segmentation API",
    description=(
        "Inference service for detecting and segmenting dolphin dorsal fins.\n\n"
        "**mask** → primary output (polygon contour)\n"
        "**bbox** → derived metadata (cv2.boundingRect of mask)\n"
        "**crop** → mask-shaped crop with transparent background (NOT bbox crop)\n\n"
        "⚠️ Current model is a pretrained COCO baseline. "
        "Replace with custom-trained checkpoint after Assignment 2/3."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    """Liveness + readiness probe."""
    loaded = is_model_loaded()
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_name=MODEL_NAME,
        note=(
            "⚠️ PLACEHOLDER: using pretrained COCO model. "
            "Replace MODEL_PATH with custom dolphin-fin checkpoint."
        ),
    )


# ---------------------------------------------------------------------------
# Predict — multipart/form-data (file upload)
# ---------------------------------------------------------------------------

@app.post(
    "/predict/upload",
    response_model=PredictResponse,
    tags=["inference"],
    summary="Predict from uploaded image file (multipart)",
)
async def predict_upload(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, …)"),
    image_id: str = Form(None, description="Optional image identifier"),
    conf_threshold: float = Form(0.25, ge=0.0, le=1.0),
    include_crop: bool = Form(True),
):
    """
    Accept an image file via multipart upload and return fin segmentation results.

    Response fields:
    - **fins[].mask**  — polygon contour (PRIMARY output)
    - **fins[].bbox**  — bounding rect DERIVED from mask (metadata only)
    - **fins[].crop_base64** — mask-shaped PNG crop, transparent outside fin
    - **fins[].crop_type**   — always "mask_shaped" (not a bbox crop)
    """
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file upload")

    img_id = image_id or str(uuid.uuid4())
    return await _run_and_build_response(image_bytes, img_id, conf_threshold, include_crop)


# ---------------------------------------------------------------------------
# Predict — JSON / base64
# ---------------------------------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["inference"],
    summary="Predict from base64-encoded image (JSON body)",
)
async def predict_base64(body: PredictRequest):
    """
    Accept a base64-encoded image in a JSON body and return fin segmentation results.

    Response fields:
    - **fins[].mask**  — polygon contour (PRIMARY output)
    - **fins[].bbox**  — bounding rect DERIVED from mask (metadata only)
    - **fins[].crop_base64** — mask-shaped PNG crop, transparent outside fin
    - **fins[].crop_type**   — always "mask_shaped" (not a bbox crop)
    """
    try:
        image_bytes = base64.b64decode(body.image_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 encoding in image_base64")

    img_id = body.image_id or str(uuid.uuid4())
    return await _run_and_build_response(
        image_bytes, img_id, body.conf_threshold, body.include_crop
    )


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

async def _run_and_build_response(
    image_bytes: bytes,
    image_id: str,
    conf_threshold: float,
    include_crop: bool,
) -> PredictResponse:
    if not is_model_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded — check /health")

    try:
        fins, inference_ms, img_w, img_h = run_inference(
            image_bytes=image_bytes,
            image_id=image_id,
            conf_threshold=conf_threshold,
            include_crop=include_crop,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Inference failed for image_id=%s", image_id)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    return PredictResponse(
        image_id=image_id,
        fins=fins,
        num_fins=len(fins),
        inference_time_ms=round(inference_ms, 2),
        model_info={
            "model_name": MODEL_NAME,
            "is_placeholder": True,
            # ⚠️ PLACEHOLDER flag — set to False once custom model is loaded
            "note": (
                "Pretrained COCO baseline. Replace MODEL_PATH env var with "
                "custom-trained dolphin-fin checkpoint after Assignment 2/3."
            ),
        },
        image_width=img_w,
        image_height=img_h,
    )
