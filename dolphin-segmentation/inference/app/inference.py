"""
Inference engine for dolphin fin instance segmentation.

# ============================================================
# ⚠️  PLACEHOLDER MODEL — PRETRAINED YOLO BASELINE
# ============================================================
# This module currently uses a pretrained YOLOv8n-seg model
# trained on COCO (80 general classes). It is used ONLY to
# demonstrate the API contract and data flow.
#
# TODO (Assignment 2/3): Replace MODEL_PATH / MODEL_NAME with
# the custom-trained checkpoint once fine-tuning on the
# dolphin-fin dataset is complete.
#
# Expected replacement:
#   MODEL_PATH = "/weights/dolphin_fin_yolo_seg_v1.pt"
#   The model should be trained on YOLO segmentation format
#   annotations exported from Roboflow (as described in design §3).
# ============================================================
#
# Core design contract (from high-level design §2 Motivation):
#
#   mask  → PRIMARY output. Polygon contour from the segmentation model.
#   bbox  → DERIVED metadata. cv2.boundingRect() of mask contour.
#           NOT used for cropping.
#   crop  → mask-shaped crop: binary mask applied as alpha channel to the
#           original image, then tightly cropped to mask bbox bounds.
#           Pixels outside the mask are TRANSPARENT (alpha=0).
#           This removes water/body/neighbours — impossible with bbox crops.
"""

import base64
import io
import logging
import os
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .schemas import BBox, FinInstance, MaskPolygon

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

# ⚠️ PLACEHOLDER: pretrained COCO model — replace with custom checkpoint
MODEL_NAME = os.getenv("MODEL_NAME", "yolov8n-seg.pt")
# When using a custom-trained checkpoint, set MODEL_PATH to the file path:
#   MODEL_PATH = "/weights/dolphin_fin_yolo_seg_v1.pt"
# and update MODEL_NAME to something like "dolphin_fin_yolo_seg_v1".
MODEL_PATH = os.getenv("MODEL_PATH", MODEL_NAME)  # falls back to auto-download

# Class filter: for the pretrained COCO model we keep all classes (None).
# ⚠️ PLACEHOLDER: once the custom model is loaded, set to None or [0]
#    (custom model likely has a single "fin" class at index 0).
FILTER_CLASSES: Optional[List[int]] = None

# Device: "cpu" or "cuda" (auto if MODEL_DEVICE not set)
DEVICE = os.getenv("MODEL_DEVICE", "cpu")


# ---------------------------------------------------------------------------
# Singleton loader
# ---------------------------------------------------------------------------

_model = None


def get_model():
    """Return (and lazily load) the YOLO segmentation model."""
    global _model
    if _model is None:
        from ultralytics import YOLO  # deferred to avoid import-time cost

        logger.info("Loading YOLO segmentation model: %s (device=%s)", MODEL_PATH, DEVICE)
        _model = YOLO(MODEL_PATH)
        logger.info("Model loaded successfully")
    return _model


def is_model_loaded() -> bool:
    return _model is not None


# ---------------------------------------------------------------------------
# Core inference logic
# ---------------------------------------------------------------------------

def decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes to a BGR numpy array."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image — unsupported format or corrupted bytes")
    return img


def mask_to_polygon(mask_xy: np.ndarray) -> List[float]:
    """Convert a (N, 2) float array of contour points to a flat list [x0, y0, x1, y1, ...]."""
    return mask_xy.flatten().tolist()


def bbox_from_contour(contour_points: np.ndarray, img_h: int, img_w: int) -> BBox:
    """
    Compute axis-aligned bounding rect from mask contour points.

    Uses cv2.boundingRect() — this is the ONLY role of bbox in this pipeline.
    The bbox is DERIVED from the mask, NOT the other way around.

    Args:
        contour_points: (N, 2) array of (x, y) absolute-pixel points
        img_h, img_w:   original image dimensions (for clipping)
    Returns:
        BBox with x, y, width, height in absolute pixels
    """
    pts = contour_points.astype(np.float32)
    pts_int = pts.reshape((-1, 1, 2)).astype(np.int32)
    x, y, w, h = cv2.boundingRect(pts_int)
    # Clip to image bounds
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = min(w, img_w - x)
    h = min(h, img_h - y)
    return BBox(x=x, y=y, width=w, height=h)


def make_mask_shaped_crop(
    image_bgr: np.ndarray,
    contour_points: np.ndarray,
) -> str:
    """
    Create a mask-shaped crop of the image.

    Workflow:
      1. Rasterize the polygon contour to a binary mask (same size as image).
      2. Apply binary mask as alpha channel to the original image
         (pixels OUTSIDE mask become transparent α=0).
      3. Compute tight crop bounds directly from the nonzero pixels of the
         rasterised mask — this is independent of bbox and ensures the crop
         boundary is driven purely by the mask shape.
      4. Encode as PNG with alpha channel and return as base64.

    Strict contract (design §2):
      - bbox is ONLY derived metadata for UI/filtering.
      - The crop tight-crop bounds come from the mask itself (np.where on the
        binary mask), NOT from the pre-computed BBox object.
      - This guarantees the two concepts cannot be confused even if bbox
        computation changes in the future.

    Args:
        image_bgr:      Original BGR image (H, W, 3)
        contour_points: (N, 2) float array of contour points (absolute px)
    Returns:
        base64-encoded PNG string (RGBA, transparent background outside mask)
    """
    h, w = image_bgr.shape[:2]

    # Step 1: rasterize polygon to binary mask
    binary_mask = np.zeros((h, w), dtype=np.uint8)
    pts_int = contour_points.reshape((-1, 1, 2)).astype(np.int32)
    cv2.fillPoly(binary_mask, [pts_int], color=255)

    # Step 2: convert image to RGBA and apply mask as alpha
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_rgb = Image.fromarray(image_rgb)
    pil_rgba = pil_rgb.convert("RGBA")

    alpha_channel = Image.fromarray(binary_mask, mode="L")
    pil_rgba.putalpha(alpha_channel)

    # Step 3: compute tight crop bounds from mask nonzero pixels
    #         (independent of bbox — bbox is metadata only)
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    if not rows.any() or not cols.any():
        raise ValueError("Rasterized mask is empty — cannot crop")
    y_idx = np.where(rows)[0]
    x_idx = np.where(cols)[0]    
    y_min = int(y_idx[0])
    y_max = int(y_idx[-1])
    x_min = int(x_idx[0])
    x_max = int(x_idx[-1])
    pil_crop = pil_rgba.crop((x_min, y_min, x_max + 1, y_max + 1))

    # Step 4: encode to base64 PNG
    buf = io.BytesIO()
    pil_crop.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def run_inference(
    image_bytes: bytes,
    image_id: str,
    conf_threshold: float = 0.25,
    include_crop: bool = True,
) -> Tuple[List[FinInstance], float, int, int]:
    """
    Run YOLO segmentation inference on raw image bytes.

    Returns:
        (fin_instances, inference_time_ms, image_width, image_height)
    """
    model = get_model()
    image_bgr = decode_image(image_bytes)
    img_h, img_w = image_bgr.shape[:2]

    t0 = time.perf_counter()
    results = model.predict(
        source=image_bgr,
        conf=conf_threshold,
        classes=FILTER_CLASSES,
        device=DEVICE,
        verbose=False,
    )
    inference_ms = (time.perf_counter() - t0) * 1000

    fins: List[FinInstance] = []

    if not results or results[0].masks is None:
        return fins, inference_ms, img_w, img_h

    result = results[0]
    masks_xy = result.masks.xy       # list of (N_i, 2) arrays, absolute px
    confidences = result.boxes.conf  # (K,) tensor

    for idx, (contour, conf_tensor) in enumerate(zip(masks_xy, confidences)):
        conf = float(conf_tensor)

        if len(contour) < 3:
            logger.debug("Skipping instance %d — polygon has < 3 points", idx)
            continue

        # PRIMARY output: mask polygon
        mask_polygon = MaskPolygon(
            points=mask_to_polygon(contour),
            width=img_w,
            height=img_h,
        )

        # DERIVED metadata: bbox from mask contour
        bbox = bbox_from_contour(contour, img_h, img_w)

        # mask-shaped crop (optional, can be large)
        # Note: bbox is NOT passed — crop bounds come from the mask itself
        crop_b64: Optional[str] = None
        if include_crop:
            try:
                crop_b64 = make_mask_shaped_crop(image_bgr, contour)
            except Exception as exc:
                #logger.warning("Failed to generate crop for instance %d: %s", idx, exc)
                logger.exception("Failed to generate crop for instance %d", idx)

        fins.append(
            FinInstance(
                instance_id=idx,
                confidence=conf,
                mask=mask_polygon,
                bbox=bbox,
                crop_base64=crop_b64,
                crop_type="mask_shaped",
            )
        )

    return fins, inference_ms, img_w, img_h
