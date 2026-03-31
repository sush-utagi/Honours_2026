"""Contextual cropping of MS COCO 2017 to single-object 512x512 crops.

For each annotation, this script centers a dynamic window on the object,
mean-pads if the crop runs off the image border, saves the crop, and writes
COCO-format annotations with exactly one bounding box per output image.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm


WINDOW = 512
BBOX_BUFFER = 0.1
ZOOM_MARGIN = 1.5  # crop side = max(bbox width/height) * ZOOM_MARGIN


def _contextual_crop(
    image: np.ndarray,
    bbox: List[float],
    window: int = WINDOW,
    buffer: float = BBOX_BUFFER,
    zoom_margin: float = ZOOM_MARGIN,
) -> Tuple[np.ndarray, List[float]]:
    """Return a padded crop centered on the object and the remapped bbox.

    Dynamic crop side = max(bbox width, bbox height) * zoom_margin (falls back
    gracefully for tiny or oversized objects). The crop is mean-padded if it
    extends beyond the image, resized to `window` (default 512), and the bbox is
    scaled and expanded by `buffer`.
    """

    h, w, _ = image.shape
    x, y, bw, bh = bbox

    # Center and dynamic crop size.
    cx = x + bw / 2.0
    cy = y + bh / 2.0
    dyn_side = max(bw, bh) * max(zoom_margin, 1e-6)
    if dyn_side <= 0:
        dyn_side = min(h, w)
    # If requested window exceeds the image, cap to the largest reasonable square.
    dyn_side = max(2.0, dyn_side)
    eff_side = int(math.ceil(dyn_side))

    half = eff_side / 2.0
    left = int(math.floor(cx - half))
    top = int(math.floor(cy - half))
    right = left + eff_side
    bottom = top + eff_side

    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - w)
    pad_bottom = max(0, bottom - h)

    # Mean padding to avoid harsh borders.
    pad_value = np.round(image.reshape(-1, image.shape[2]).mean(axis=0)).astype(image.dtype)
    padded = np.empty(
        (h + pad_top + pad_bottom, w + pad_left + pad_right, image.shape[2]),
        dtype=image.dtype,
    )
    padded[...] = pad_value
    padded[pad_top : pad_top + h, pad_left : pad_left + w] = image

    # Shift crop window into padded coordinates.
    top_p = top + pad_top
    left_p = left + pad_left
    crop = padded[top_p : top_p + eff_side, left_p : left_p + eff_side]

    # Bbox in crop coordinates (pre-buffer, pre-resize).
    x1 = x - left
    y1 = y - top
    x2 = x1 + bw
    y2 = y1 + bh

    # Resize crop to fixed output window.
    scale = window / float(eff_side)
    crop_img = Image.fromarray(crop).resize((window, window), resample=Image.BILINEAR)
    crop = np.array(crop_img, dtype=image.dtype)

    # Scale bbox to output size.
    x1 *= scale
    y1 *= scale
    x2 *= scale
    y2 *= scale

    # Apply buffer on scaled bbox.
    bw_scaled = x2 - x1
    bh_scaled = y2 - y1
    x1_buf = x1 - bw_scaled * buffer
    y1_buf = y1 - bh_scaled * buffer
    x2_buf = x2 + bw_scaled * buffer
    y2_buf = y2 + bh_scaled * buffer

    # Clamp to output window.
    x1_cl = max(0.0, min(float(window), x1_buf))
    y1_cl = max(0.0, min(float(window), y1_buf))
    x2_cl = max(0.0, min(float(window), x2_buf))
    y2_cl = max(0.0, min(float(window), y2_buf))

    w_new = max(0.0, x2_cl - x1_cl)
    h_new = max(0.0, y2_cl - y1_cl)

    return crop, [x1_cl, y1_cl, w_new, h_new]


def _process_split(
    split: str,
    input_root: Path,
    output_root: Path,
    window: int = WINDOW,
    zoom_margin: float = ZOOM_MARGIN,
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

            crop_np, new_bbox = _contextual_crop(
                img_np, bbox, window=window, buffer=BBOX_BUFFER, zoom_margin=zoom_margin
            )
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


def _repartition_train_val(
    output_root: Path,
    val_ratio: float,
    seed: int,
) -> None:
    """Create a stratified val split from train and promote old val to test.

    - Reads contextual crops: single_instances_train.json / single_instances_val.json
    - Moves old val images -> images/test and writes single_instances_test.json
    - Stratifies the train split per category to carve out a new val portion,
      preserving class balance; moves those images to images/val and rewrites
      single_instances_train.json / single_instances_val.json with fresh ids.
    """

    train_ann_path = output_root / "annotations" / "single_instances_train.json"
    val_ann_path = output_root / "annotations" / "single_instances_val.json"

    if not train_ann_path.exists() or not val_ann_path.exists():
        print("[warn] Cannot repartition: missing contextual train/val annotations.")
        return

    with open(train_ann_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(val_ann_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    categories = train_data.get("categories", [])
    licenses = train_data.get("licenses", [])

    train_dir = output_root / "images" / "train"
    val_dir = output_root / "images" / "val"
    test_dir = output_root / "images" / "test"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Promote old val -> test
    test_images = []
    test_annotations = []
    next_img_id = 1
    next_ann_id = 1
    img_map_val = {img["id"]: img for img in val_data.get("images", [])}
    for ann in val_data.get("annotations", []):
        img = img_map_val.get(ann["image_id"])
        if not img:
            continue
        fname = img["file_name"]
        src = val_dir / fname
        dst = test_dir / fname
        if src.exists():
            shutil.move(src, dst)
        new_img = {"id": next_img_id, "file_name": fname, "width": img["width"], "height": img["height"]}
        new_ann = {
            "id": next_ann_id,
            "image_id": next_img_id,
            "category_id": ann["category_id"],
            "bbox": ann["bbox"],
            "area": ann.get("area", 0.0),
            "iscrowd": ann.get("iscrowd", 0),
            "segmentation": [],
        }
        test_images.append(new_img)
        test_annotations.append(new_ann)
        next_img_id += 1
        next_ann_id += 1

    test_json = {
        "info": {"description": "Contextual crops (test from original val)", "version": "1.0"},
        "licenses": licenses,
        "categories": categories,
        "images": test_images,
        "annotations": test_annotations,
    }
    with open(output_root / "annotations" / "single_instances_test.json", "w", encoding="utf-8") as f:
        json.dump(test_json, f)

    # Stratified carve-out: train -> new val
    rng = random.Random(seed)
    img_map_train = {img["id"]: img for img in train_data.get("images", [])}
    by_cat: Dict[int, List[Tuple[Dict, Dict]]] = {}
    for ann in train_data.get("annotations", []):
        img = img_map_train.get(ann["image_id"])
        if img is None:
            continue
        by_cat.setdefault(ann["category_id"], []).append((img, ann))

    selected_val: List[Tuple[Dict, Dict]] = []
    remaining_train: List[Tuple[Dict, Dict]] = []
    for cat, items in by_cat.items():
        rng.shuffle(items)
        val_count = int(round(len(items) * val_ratio))
        if val_count == 0 and len(items) > 0:
            val_count = 1  # guarantee at least one when available
        selected_val.extend(items[:val_count])
        remaining_train.extend(items[val_count:])

    # Move files and reindex
    def _reindex_and_move(pairs: List[Tuple[Dict, Dict]], dest_dir: Path, start_img_id: int = 1, start_ann_id: int = 1):
        images_out = []
        annotations_out = []
        img_id = start_img_id
        ann_id = start_ann_id
        for img, ann in pairs:
            fname = img["file_name"]
            src = train_dir / fname
            dst = dest_dir / fname
            if src.exists() and src != dst:
                shutil.move(src, dst)
            images_out.append({"id": img_id, "file_name": fname, "width": img["width"], "height": img["height"]})
            annotations_out.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": ann["category_id"],
                    "bbox": ann["bbox"],
                    "area": ann.get("area", 0.0),
                    "iscrowd": ann.get("iscrowd", 0),
                    "segmentation": [],
                }
            )
            img_id += 1
            ann_id += 1
        return images_out, annotations_out

    # Clear and recreate val dir for new split
    if val_dir.exists():
        shutil.rmtree(val_dir)
    val_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)

    train_images, train_annotations = _reindex_and_move(remaining_train, train_dir)
    val_images, val_annotations = _reindex_and_move(selected_val, val_dir)

    train_json = {
        "info": {"description": "Contextual crops (train, rebalanced)", "version": "1.0"},
        "licenses": licenses,
        "categories": categories,
        "images": train_images,
        "annotations": train_annotations,
    }
    val_json = {
        "info": {"description": "Contextual crops (val from train, stratified)", "version": "1.0"},
        "licenses": licenses,
        "categories": categories,
        "images": val_images,
        "annotations": val_annotations,
    }

    with open(train_ann_path, "w", encoding="utf-8") as f:
        json.dump(train_json, f)
    with open(val_ann_path, "w", encoding="utf-8") as f:
        json.dump(val_json, f)

    print(
        f"[repartition] train images: {len(train_images)}, val images: {len(val_images)}, "
        f"test images (from old val): {len(test_images)}"
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
        "--zoom-margin",
        default=ZOOM_MARGIN,
        type=float,
        help="Dynamic crop side = max(bbox w,h) * zoom_margin (e.g., 1.5)",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["train", "val"],
        help="Dataset splits to process",
    )
    parser.add_argument(
        "--stratify-train-to-val",
        action="store_true",
        help="Create a new val split stratified from train, and promote original val to test.",
    )
    parser.add_argument(
        "--val-ratio-from-train",
        type=float,
        default=0.1,
        help="Fraction of train (per class) to use for the new val split (default: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stratified splitting (default: 42).",
    )

    args = parser.parse_args()

    for split in args.splits:
        _process_split(split, args.input_root, args.output_root, args.window, args.zoom_margin)
    if args.stratify_train_to_val:
        _repartition_train_val(args.output_root, args.val_ratio_from_train, args.seed)


if __name__ == "__main__":
    main()
