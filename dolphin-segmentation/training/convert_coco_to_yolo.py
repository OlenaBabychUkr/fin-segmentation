import json
import os
from pathlib import Path

from pycocotools import mask as mask_utils
import cv2
import numpy as np


SOURCE_ROOT = Path("training/dataset_coco")
OUTPUT_ROOT = Path("training/dataset_yolo")

SPLITS = {
    "train": "train",
    "valid": "val",
    "test": "test",
}


def rle_to_polygon(segmentation, height, width):
    """
    Convert COCO RLE segmentation to polygon points.
    Returns a list of polygons.
    """

    if isinstance(segmentation, dict):
        rle = segmentation

        if isinstance(rle["counts"], str):
            rle = {
                "size": rle["size"],
                "counts": rle["counts"].encode("utf-8"),
            }

        mask = mask_utils.decode(rle)

    elif isinstance(segmentation, list):
        # Already polygon format
        return segmentation

    else:
        return []

    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    polygons = []

    for contour in contours:
        if len(contour) < 3:
            continue

        contour = contour.reshape(-1, 2)

        polygon = []

        for x, y in contour:
            polygon.extend([
                x / width,
                y / height,
            ])

        polygons.append(polygon)

    return polygons


def convert_split(source_split, output_split):
    source_dir = SOURCE_ROOT / source_split
    annotation_file = source_dir / "_annotations.coco.json"

    if not annotation_file.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {annotation_file}"
        )

    with open(annotation_file, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images_by_id = {
        img["id"]: img
        for img in coco["images"]
    }

    annotations_by_image = {}

    for ann in coco["annotations"]:
        annotations_by_image.setdefault(
            ann["image_id"],
            []
        ).append(ann)

    output_images = OUTPUT_ROOT / "images" / output_split
    output_labels = OUTPUT_ROOT / "labels" / output_split

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    for image_id, image_info in images_by_id.items():

        filename = image_info["file_name"]
        width = image_info["width"]
        height = image_info["height"]

        source_image = source_dir / filename
        target_image = output_images / filename

        if source_image.exists():
            target_image.write_bytes(
                source_image.read_bytes()
            )

        label_path = (
            output_labels /
            f"{Path(filename).stem}.txt"
        )

        lines = []

        for ann in annotations_by_image.get(image_id, []):

            category_id = ann["category_id"]

            # Roboflow COCO has dorsal fin = category 1.
            # YOLO class indices must start from 0.
            class_id = category_id - 1

            polygons = rle_to_polygon(
                ann["segmentation"],
                height,
                width,
            )

            for polygon in polygons:

                if len(polygon) < 6:
                    continue

                coords = " ".join(
                    f"{value:.6f}"
                    for value in polygon
                )

                lines.append(
                    f"{class_id} {coords}"
                )

        label_path.write_text(
            "\n".join(lines),
            encoding="utf-8",
        )

    print(
        f"{source_split}: "
        f"{len(images_by_id)} images converted"
    )


def create_data_yaml():
    yaml_content = """path: /data
train: images/train
val: images/val
test: images/test

names:
  0: dorsal_fin
"""

    (OUTPUT_ROOT / "data.yaml").write_text(
        yaml_content,
        encoding="utf-8",
    )


def main():

    for source_split, output_split in SPLITS.items():
        convert_split(
            source_split,
            output_split,
        )

    create_data_yaml()

    print()
    print("Conversion complete.")
    print(f"Dataset saved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()