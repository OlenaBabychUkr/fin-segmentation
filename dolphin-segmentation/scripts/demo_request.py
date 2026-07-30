#!/usr/bin/env python3
"""
Demo: send a test image to the inference API and print the result.

Usage:
    # With a local image file:
    python scripts/demo_request.py --image path/to/dolphin.jpg

    # Download a free test image automatically:
    python scripts/demo_request.py --download-test-image

    # Adjust confidence threshold:
    python scripts/demo_request.py --image dolphin.jpg --conf 0.3

    # Skip mask-shaped crops in response (faster):
    python scripts/demo_request.py --image dolphin.jpg --no-crop

Requirements:
    pip install requests Pillow

⚠️ The pretrained YOLO model is trained on COCO (80 general classes, NOT fins).
   It likely will NOT detect dolphin fins specifically. This demo just proves
   the API contract is working correctly. Replace MODEL_PATH with a custom
   checkpoint after Assignment 2/3.
"""

import argparse
import base64
import json
import sys
import urllib.request
from pathlib import Path

import requests

API_BASE = "http://localhost:8000"

# A freely usable public domain dolphin image for quick testing
TEST_IMAGE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/10/"
    "Tursiops_truncatus_01.jpg/640px-Tursiops_truncatus_01.jpg"
)
TEST_IMAGE_PATH = Path("test_dolphin.jpg")


def download_test_image():
    if TEST_IMAGE_PATH.exists():
        print(f"Using cached test image: {TEST_IMAGE_PATH}")
        return TEST_IMAGE_PATH
    print(f"Downloading test image from Wikipedia …")
    urllib.request.urlretrieve(TEST_IMAGE_URL, TEST_IMAGE_PATH)
    print(f"Saved to {TEST_IMAGE_PATH}")
    return TEST_IMAGE_PATH


def check_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=10)
        r.raise_for_status()
        print("=== /health ===")
        print(json.dumps(r.json(), indent=2))
        print()
        return r.json().get("model_loaded", False)
    except Exception as exc:
        print(f"Health check failed: {exc}")
        print("Is the API running?  Try:  docker-compose up inference")
        sys.exit(1)


def predict_with_upload(image_path: Path, conf: float, include_crop: bool):
    """Use multipart file upload endpoint."""
    print(f"=== POST /predict/upload  (file: {image_path}) ===")
    with open(image_path, "rb") as f:
        r = requests.post(
            f"{API_BASE}/predict/upload",
            files={"file": (image_path.name, f, "image/jpeg")},
            data={
                "conf_threshold": str(conf),
                "include_crop": "true" if include_crop else "false",
            },
            timeout=120,
        )
    r.raise_for_status()
    return r.json()


def predict_with_base64(image_path: Path, conf: float, include_crop: bool):
    """Use JSON / base64 endpoint."""
    print(f"=== POST /predict  (base64, file: {image_path}) ===")
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    payload = {
        "image_base64": b64,
        "image_id": image_path.stem,
        "conf_threshold": conf,
        "include_crop": include_crop,
    }
    r = requests.post(f"{API_BASE}/predict", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def print_result(result: dict):
    """Print response summary without flooding the terminal with crop bytes."""
    r = dict(result)
    for fin in r.get("fins", []):
        if fin.get("crop_base64"):
            # Truncate for display; actual bytes are fully present
            fin["crop_base64"] = fin["crop_base64"][:60] + "…[base64 PNG]"
    print(json.dumps(r, indent=2))


def save_example_response(result: dict, path: Path = Path("example_response.json")):
    """Save full response to file for reference."""
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nFull response saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Demo: dolphin fin segmentation API")
    parser.add_argument("--image", type=Path, help="Path to a local image file")
    parser.add_argument("--download-test-image", action="store_true",
                        help="Download a test dolphin image automatically")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--no-crop", action="store_true",
                        help="Skip mask-shaped crop generation (faster)")
    parser.add_argument("--base64", action="store_true",
                        help="Use JSON/base64 endpoint instead of multipart upload")
    args = parser.parse_args()

    # Determine image path
    if args.download_test_image or (not args.image):
        image_path = download_test_image()
    else:
        image_path = args.image
        if not image_path.exists():
            print(f"File not found: {image_path}")
            sys.exit(1)

    # Health check
    check_health()

    # Run prediction
    include_crop = not args.no_crop
    if args.base64:
        result = predict_with_base64(image_path, args.conf, include_crop)
    else:
        result = predict_with_upload(image_path, args.conf, include_crop)

    print_result(result)
    save_example_response(result)

    print(f"\n✓ Detected {result['num_fins']} fin instance(s) "
          f"in {result['inference_time_ms']:.1f} ms")
    print(f"  model: {result['model_info']['model_name']}")
    if result["model_info"].get("is_placeholder"):
        print("  ⚠️  PLACEHOLDER model — pretrained COCO, not dolphin-fin trained.")
        print("     Replace MODEL_PATH with custom checkpoint (Assignment 2/3).")

    if result["fins"]:
        print("\nFirst fin instance:")
        fin = result["fins"][0]
        print(f"  confidence  : {fin['confidence']:.3f}")
        print(f"  bbox        : {fin['bbox']} (derived from mask)")
        print(f"  mask points : {len(fin['mask']['points']) // 2} polygon points")
        print(f"  crop_type   : {fin['crop_type']}")


if __name__ == "__main__":
    main()
