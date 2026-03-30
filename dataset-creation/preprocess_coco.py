"""Contextual cropping of MS COCO 2017 to single-object 512x512 crops.

For each annotation, this script centers a 512x512 window on the object,
zero-pads if the window runs off the image border, saves the crop, and writes
COCO-format annotations with exactly one bounding box per output image.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm


WINDOW = 512


def _contextual_crop(
    image: np.ndarray, bbox: List[float], window: int = WINDOW
) -> Tuple[np.ndarray, List[float]]:
    """Return padded 512x512 crop centered on the bbox and the remapped bbox.

    The crop origin is the top-left of the window; padding is filled with zeros.
    Bounding box is clamped to the crop bounds.
    """

    h, w, _ = image.shape
    x, y, bw, bh = bbox

    cx = x + bw / 2.0
    cy = y + bh / 2.0
    half = window // 2

    left = int(math.floor(cx - half))
    top = int(math.floor(cy - half))
    right = left + window
    bottom = top + window

    pad_left = max(0, -left)
    pad_top = max(0, -top)

    crop = np.zeros((window, window, 3), dtype=np.uint8)

    src_x1 = max(0, left)
    src_y1 = max(0, top)
    src_x2 = min(w, right)
    src_y2 = min(h, bottom)

    dst_x1 = pad_left
    dst_y1 = pad_top
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    crop[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]

    x1_new = max(0.0, min(float(window), x - left))
    y1_new = max(0.0, min(float(window), y - top))
    x2_new = max(0.0, min(float(window), x + bw - left))
    y2_new = max(0.0, min(float(window), y + bh - top))
    w_new = max(0.0, x2_new - x1_new)
    h_new = max(0.0, y2_new - y1_new)

    return crop, [x1_new, y1_new, w_new, h_new]


def _process_split(
    split: str,
    input_root: Path,
    output_root: Path,
    window: int = WINDOW,
) -> None:
    ann_path = input_root / "annotations" / f"instances_{split}.json"
    if not ann_path.exists():
        print(f"[skip] No annotations found for split '{split}' at {ann_path}")
        return

    coco = COCO(str(ann_path))
    ann_ids_all = coco.getAnnIds()
    if not ann_ids_all:
        print(f"[skip] No annotations in '{split}' (likely the test split).")
        return

    img_dir = input_root / "images" / split
    out_img_dir = output_root / "images" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_ann_path = output_root / "annotations" / f"single_instances_{split}.json"
    out_ann_path.parent.mkdir(parents=True, exist_ok=True)

    categories = coco.dataset.get("categories", [])
    licenses = coco.dataset.get("licenses", [])

    images_out: List[Dict] = []
    annotations_out: List[Dict] = []
    next_image_id = 1
    next_ann_id = 1

    for img_id in tqdm(coco.getImgIds(), desc=f"{split} images"):
        img_info = coco.loadImgs([img_id])[0]
        file_name = img_info["file_name"]
        path = img_dir / file_name
        if not path.exists():
            continue

        img = Image.open(path).convert("RGB")
        img_np = np.array(img)

        ann_ids = coco.getAnnIds(imgIds=[img_id])
        for ann_id in ann_ids:
            ann = coco.loadAnns([ann_id])[0]
            bbox = ann.get("bbox", [0, 0, 0, 0])

            crop_np, new_bbox = _contextual_crop(img_np, bbox, window)
            if new_bbox[2] <= 0 or new_bbox[3] <= 0:
                continue

            new_file_name = f"{img_id}_{ann_id}.jpg"
            Image.fromarray(crop_np).save(out_img_dir / new_file_name, quality=95)

            images_out.append(
                {
                    "id": next_image_id,
                    "file_name": new_file_name,
                    "width": window,
                    "height": window,
                }
            )

            annotations_out.append(
                {
                    "id": next_ann_id,
                    "image_id": next_image_id,
                    "category_id": ann["category_id"],
                    "bbox": new_bbox,
                    "area": float(new_bbox[2] * new_bbox[3]),
                    "iscrowd": ann.get("iscrowd", 0),
                    "segmentation": [],
                }
            )

            next_image_id += 1
            next_ann_id += 1

    output_json = {
        "info": {
            "description": "COCO 2017 contextual crops (single object)",
            "version": "1.0",
        },
        "licenses": licenses,
        "categories": categories,
        "images": images_out,
        "annotations": annotations_out,
    }

    with open(out_ann_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f)

    print(
        f"[done] {split}: wrote {len(images_out)} crops to {out_img_dir} and annotations to {out_ann_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Contextual crops from COCO annotations")
    parser.add_argument(
        "--input-root",
        default="coco_dataset/split",
        type=Path,
        help="Root containing images/<split> and annotations/instances_<split>.json",
    )
    parser.add_argument(
        "--output-root",
        default="coco_dataset/contextual_crops",
        type=Path,
        help="Root to write cropped images and single-instance annotations",
    )
    parser.add_argument(
        "--window",
        default=WINDOW,
        type=int,
        help="Crop window size (pixels); must be even",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["train", "val"],
        help="Dataset splits to process",
    )

    args = parser.parse_args()

    for split in args.splits:
        _process_split(split, args.input_root, args.output_root, args.window)


if __name__ == "__main__":
    main()
