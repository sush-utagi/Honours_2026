"""Remove synthetic injections from the COCO contextual crops dataset.

Usage:
    python scripts/cleanse_training_set.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser(description="Cleanse synthetic injections from dataset.")
    parser.add_argument("--annotations", default="coco_dataset/contextual_crops/annotations/single_instances_train.json")
    parser.add_argument("--images-dir", default="coco_dataset/contextual_crops/images/train")
    parser.add_argument("--identifier", default="_syn_", help="Substring identifying injected files.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would happen without making changes.")
    args = parser.parse_args()

    ann_path = Path(args.annotations)
    images_dir = Path(args.images_dir)
    
    if not ann_path.exists() or not images_dir.exists():
        print("[error] Paths do not exist.")
        return

    print(f"[init] Loading annotations from {ann_path}...")
    with ann_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    original_img_count = len(data.get("images", []))
    original_ann_count = len(data.get("annotations", []))

    # 1. Filter out images with identifier in filename
    synthetic_image_ids = set()
    clean_images = []
    
    for img in data.get("images", []):
        if args.identifier in img["file_name"]:
            synthetic_image_ids.add(img["id"])
        else:
            clean_images.append(img)
            
    # 2. Filter out annotations linked to synthetic images
    clean_annotations = [
        ann for ann in data.get("annotations", []) 
        if ann["image_id"] not in synthetic_image_ids
    ]
    
    removed_img_count = original_img_count - len(clean_images)
    removed_ann_count = original_ann_count - len(clean_annotations)
    
    print(f"[info] Identified {removed_img_count} synthetic image records and {removed_ann_count} annotation records.")

    # 3. Clean files from filesystem
    removed_files = 0
    if images_dir.exists():
        for img_path in images_dir.iterdir():
            if img_path.is_file() and args.identifier in img_path.name:
                if not args.dry_run:
                    img_path.unlink()
                removed_files += 1

    print(f"[info] Identified {removed_files} synthetic image files on disk.")

    if args.dry_run:
        print("[dry-run] No changes made.")
        return

    # 4. Save cleaned annotations
    if removed_img_count > 0 or removed_files > 0:
        data["images"] = clean_images
        data["annotations"] = clean_annotations
        
        with ann_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"[success] Dataset cleansed. Restored to {len(clean_images)} images and {len(clean_annotations)} annotations.")
    else:
        print(f"[success] Dataset is already clean.")

if __name__ == "__main__":
    main()