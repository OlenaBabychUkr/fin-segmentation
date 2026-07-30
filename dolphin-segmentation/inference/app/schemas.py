"""
Pydantic schemas for the Dolphin Fin Instance Segmentation API.

Design note (from high-level design §2 Motivation):
  - `mask`  → PRIMARY output: polygon contour points extracted from the
              segmentation mask. This is WHY we use instance segmentation
              instead of object detection.
  - `bbox`  → DERIVED metadata only: computed via cv2.boundingRect() from
              the mask contour. Used for UI overlays / quick filtering.
              NOT used for cropping.
  - `crop`  → mask-shaped crop: binary mask applied to the original image
              (pixels outside mask = transparent), then tightly cropped to
              mask bbox. This yields a fin silhouette, NOT a rectangular
              bbox crop.  crop_type field makes this explicit.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sub-types
# ---------------------------------------------------------------------------

class BBox(BaseModel):
    """Axis-aligned bounding rectangle DERIVED from the mask contour.

    All values in absolute pixels of the input image.
    Computed via cv2.boundingRect() — NOT used for cropping (see crop_type).
    """
    x: int = Field(..., description="Left edge (absolute px)")
    y: int = Field(..., description="Top edge (absolute px)")
    width: int = Field(..., description="Width (absolute px)")
    height: int = Field(..., description="Height (absolute px)")


class MaskPolygon(BaseModel):
    """Polygon contour of the segmentation mask.

    `points` is a flat list [x0, y0, x1, y1, ...] in absolute pixels of the
    input image (same coordinate system as bbox).

    For downstream use, the polygon can be rasterized back to a binary mask
    via cv2.fillPoly() or shapely.geometry.Polygon.
    """
    points: List[float] = Field(
        ...,
        description="Flat list [x0, y0, x1, y1, ...] of contour points (abs px)",
    )
    width: int = Field(..., description="Original image width (for denormalisation)")
    height: int = Field(..., description="Original image height (for denormalisation)")


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Request body for POST /predict when using JSON/base64 upload."""

    image_base64: str = Field(
        ...,
        description="Base64-encoded image bytes (JPEG, PNG, …)",
    )
    image_id: Optional[str] = Field(
        None,
        description="Optional caller-supplied identifier for this image. "
                    "If omitted the server generates a UUID.",
    )
    conf_threshold: float = Field(
        0.25,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score to include an instance in the response.",
    )
    include_crop: bool = Field(
        True,
        description="If true, each fin instance includes a mask-shaped crop "
                    "(base64 PNG with alpha channel).",
    )


# ---------------------------------------------------------------------------
# Per-instance response
# ---------------------------------------------------------------------------

class FinInstance(BaseModel):
    """A single detected fin instance."""

    instance_id: int = Field(..., description="Zero-based index within this image's results")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")

    # PRIMARY output — the reason instance segmentation was chosen over detection
    mask: MaskPolygon = Field(
        ...,
        description="Polygon mask contour — PRIMARY output of the segmentation model",
    )

    # DERIVED metadata — computed from mask, not from model directly
    bbox: BBox = Field(
        ...,
        description="Bounding rect DERIVED via cv2.boundingRect(mask_contour). "
                    "Metadata only — NOT the basis for the crop.",
    )

    # mask-shaped crop
    crop_base64: Optional[str] = Field(
        None,
        description="Base64-encoded PNG of the mask-shaped crop. "
                    "Pixels OUTSIDE the mask are transparent (alpha=0). "
                    "Image is tightly cropped to the mask's bounding rect to remove "
                    "empty border space — see crop_type for semantics.",
    )
    crop_type: str = Field(
        "mask_shaped",
        description="Always 'mask_shaped': crop is the fin silhouette with "
                    "transparent background, NOT a rectangular bbox crop. "
                    "This is the key advantage of instance segmentation (design §2).",
    )


# ---------------------------------------------------------------------------
# Top-level response
# ---------------------------------------------------------------------------

class PredictResponse(BaseModel):
    image_id: str = Field(..., description="Identifier for this image (caller-supplied or UUID)")
    fins: List[FinInstance] = Field(..., description="Detected fin instances (may be empty)")
    num_fins: int = Field(..., description="Number of fins detected (len(fins))")
    inference_time_ms: float = Field(..., description="Wall-clock inference time in milliseconds")
    model_info: dict = Field(
        ...,
        description="Info about the model used. "
                    "model_name='yolov8n-seg-pretrained' signals this is the placeholder "
                    "baseline — replace with custom-trained checkpoint after Assignment 2/3.",
    )
    image_width: int
    image_height: int


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    note: str
