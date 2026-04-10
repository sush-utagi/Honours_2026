"""Inject synthetic images into the training split and update COCO annotations.

Usage (example):
    python3 scripts/add_synthetic_to_train.py \
        --synthetic-root /path/to/synth_by_class \
        --dataset-root coco_dataset/contextual_crops/images \
        --annotations coco_dataset/contextual_crops/annotations/single_instances_train.json

Assumptions:
- Synthetic images are organised in class-named subdirectories under ``--synthetic-root``
  (e.g., ``person/``, ``bicycle/``) and filenames can be anything.
- Only the *training* split is modified; validation/test remain untouched.
- Bounding boxes are set to the full image (you can later refine if needed).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


def load_annotations(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_annotations(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def next_ids(data: Dict) -> Tuple[int, int]:
    """Return (next_image_id, next_annotation_id)."""

    max_img = max(img["id"] for img in data["images"]) if data["images"] else 0
    max_ann = max(ann["id"] for ann in data["annotations"]) if data["annotations"] else 0
    return max_img + 1, max_ann + 1


def build_category_lookup(data: Dict) -> Dict[str, int]:
    """Map category name -> id."""

    return {c["name"]: c["id"] for c in data.get("categories", [])}


def ingest_class_dir(
    class_dir: Path,
    class_name: str,
    category_id: int,
    dest_dir: Path,
    data: Dict,
    next_img_id: int,
    next_ann_id: int,
) -> Tuple[int, int, int]:
    """Copy all images from class_dir into dest_dir and add annotations.

    Returns (images_added, next_image_id, next_annotation_id).
    """

    images_added = 0
    for src in sorted(class_dir.iterdir()):
        if not src.is_file():
            continue

        # Read size
        with Image.open(src) as im:
            width, height = im.size

        # Destination filename to avoid clashes
        dest_name = f"{class_name}_syn_{next_img_id}{src.suffix.lower()}"
        dest_path = dest_dir / dest_name

        # Copy image
        shutil.copy2(src, dest_path)

        # Append image record
        data["images"].append(
            {
                "id": next_img_id,
                "file_name": dest_name,
                "width": width,
                "height": height,
            }
        )

        # Full-frame bbox
        data["annotations"].append(
            {
                "id": next_ann_id,
                "image_id": next_img_id,
                "category_id": category_id,
                "bbox": [0.0, 0.0, float(width), float(height)],
                "area": float(width * height),
                "iscrowd": 0,
                "segmentation": [],
            }
        )

        next_img_id += 1
        next_ann_id += 1
        images_added += 1

    return images_added, next_img_id, next_ann_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Add synthetic images to training split and update COCO JSON.")
    parser.add_argument("--synthetic-root", required=True, help="Root containing class-named folders of synthetic images.")
    parser.add_argument(
        "--dataset-root",
        default="coco_dataset/contextual_crops/images",
        help="Dataset images root containing train/ val/ test/ subfolders.",
    )
    parser.add_argument(
        "--annotations",
        default="coco_dataset/contextual_crops/annotations/single_instances_train.json",
        help="Path to the training annotations JSON to update.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not write files; just report what would happen.")
    args = parser.parse_args()

    synth_root = Path(args.synthetic_root)
    dataset_root = Path(args.dataset_root)
    train_dir = dataset_root / "train"
    ann_path = Path(args.annotations)

    if not train_dir.is_dir():
        raise SystemExit(f"Train directory not found: {train_dir}")
    if not ann_path.is_file():
        raise SystemExit(f"Annotations file not found: {ann_path}")
    if not synth_root.is_dir():
        raise SystemExit(f"Synthetic root not found: {synth_root}")

    data = load_annotations(ann_path)
    cat_lookup = build_category_lookup(data)

    next_img_id, next_ann_id = next_ids(data)
    total_added = 0

    for class_dir in sorted(p for p in synth_root.iterdir() if p.is_dir()):
        class_name = class_dir.name
        if class_name not in cat_lookup:
            print(f"[skip] '{class_name}' not in categories; skipping folder {class_dir}")
            continue
        category_id = cat_lookup[class_name]
        added, next_img_id, next_ann_id = ingest_class_dir(
            class_dir, class_name, category_id, train_dir, data, next_img_id, next_ann_id
        )
        total_added += added
        print(f"[ok] {class_name}: added {added} images")

    print(f"[summary] total synthetic images added to train: {total_added}")

    if args.dry_run:
        print("[dry-run] no files written; rerun without --dry-run to commit changes")
        return

    save_annotations(data, ann_path)
    print(f"[done] annotations updated at {ann_path}")


if __name__ == "__main__":
    main()
