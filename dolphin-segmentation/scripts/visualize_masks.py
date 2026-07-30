#!/usr/bin/env python3
"""
visualize_masks.py — overlay segmentation masks on the original image.

Used for manual review of inference results (design §5.5 manual review step).

Usage:
    # From API response JSON:
    python scripts/visualize_masks.py --image dolphin.jpg --response example_response.json

    # Live prediction + visualize in one step:
    python scripts/visualize_masks.py --image dolphin.jpg --live

    # Save output to file:
    python scripts/visualize_masks.py --image dolphin.jpg --response response.json --out viz.jpg

Requirements:
    pip install requests Pillow opencv-python numpy
"""

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


API_BASE = "http://localhost:8000" #"http://inference:8000" #

# Palette for coloring mask instances
COLORS = [
    (255, 56,  56),   # red
    (56,  255, 56),   # green
    (56,  56,  255),  # blue
    (255, 200, 56),   # yellow
    (200, 56,  255),  # purple
    (56,  255, 200),  # cyan-ish
]


def load_response(response_path: Path) -> dict:
    with open(response_path) as f:
        return json.load(f)


def predict_live(image_path: Path) -> dict:
    if not HAS_REQUESTS:
        print("Install requests:  pip install requests")
        sys.exit(1)
    with open(image_path, "rb") as f:
        r = requests.post(
            f"{API_BASE}/predict/upload",
            files={"file": (image_path.name, f, "image/jpeg")},
            data={"conf_threshold": "0.25", "include_crop": "false"},
            timeout=120,
        )
    r.raise_for_status()
    return r.json()


def draw_masks(image_path: Path, response: dict, out_path: Path | None = None):
    """
    Overlay mask polygons, bbox rectangles, and confidence labels on the image.

    Rendering layers (back to front):
      1. Semi-transparent filled polygon (mask)
      2. Polygon outline (contour)
      3. Bbox rectangle (dashed-style — derived metadata)
      4. Confidence + instance ID label
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Cannot open image: {image_path}")
        sys.exit(1)

    overlay = img.copy()
    fins = response.get("fins", [])

    if not fins:
        print("No fin instances in response — nothing to visualize.")
        print("(Pretrained COCO model may not detect dolphin fins.)")
    else:
        print(f"Visualizing {len(fins)} fin instance(s) …")

    for fin in fins:
        color = COLORS[fin["instance_id"] % len(COLORS)]

        # ── Filled polygon (semi-transparent) ─────────────────────────────
        pts_flat = fin["mask"]["points"]
        if len(pts_flat) >= 6:  # at least 3 points
            pts = np.array(pts_flat, dtype=np.float32).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(overlay, [pts], color=color)

        # ── Polygon outline ────────────────────────────────────────────────
        if len(pts_flat) >= 6:
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)

        # ── BBox rectangle (derived metadata, shown as dashed-ish) ────────
        b = fin["bbox"]
        cv2.rectangle(
            img,
            (b["x"], b["y"]),
            (b["x"] + b["width"], b["y"] + b["height"]),
            color=(200, 200, 200),  # grey — indicates "derived, not primary"
            thickness=1,
        )

        # ── Label ─────────────────────────────────────────────────────────
        label = f"#{fin['instance_id']}  {fin['confidence']:.2f}"
        lx, ly = b["x"], max(b["y"] - 8, 12)
        cv2.putText(
            img, label, (lx, ly),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )

    # Blend overlay (filled masks) at 40% opacity
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

    # Legend
    cv2.putText(
        img,
        f"Fins: {len(fins)}  |  grey rect = bbox (derived)  |  color fill = mask",
        (8, img.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
    )

    if out_path:
        cv2.imwrite(str(out_path), img)
        print(f"Visualization saved to {out_path}")
    else:
        # Try to display (works in local env; may not work in headless container)
        cv2.imshow("Dolphin Fin Segmentation", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img


def main():
    parser = argparse.ArgumentParser(description="Visualize dolphin fin segmentation masks")
    parser.add_argument("--image", type=Path, required=True, help="Input image path")
    parser.add_argument("--response", type=Path, help="Path to saved API response JSON")
    parser.add_argument("--live", action="store_true",
                        help="Call the API live instead of using a saved response")
    parser.add_argument("--out", type=Path, default=None,
                        help="Save visualization to this file (default: display)")
    args = parser.parse_args()

    if not args.image.exists():
        print(f"Image not found: {args.image}")
        sys.exit(1)

    if args.live:
        print("Running live prediction …")
        response = predict_live(args.image)
    elif args.response:
        response = load_response(args.response)
    else:
        print("Provide --response <json> or use --live")
        sys.exit(1)

    # Default output path
    out_path = args.out
    if out_path is None and not args.live:
        out_path = args.image.parent / (args.image.stem + "_viz.jpg")

    draw_masks(args.image, response, out_path)


if __name__ == "__main__":
    main()
