"""Check the proportion of training images with neutral gray (127) padding.

This script identifies 'toaster' and 'hair drier' images in the COCO training set
and reports how many of them contain the neutral gray padding in the corners.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

def check_padding(image_path: Path, threshold: int = 127, tolerance: int = 0) -> bool:
    """Check if any corner of the image matches the neutral gray padding."""
    try:
        with Image.open(image_path) as img:
            img_np = np.array(img.convert("RGB"))
            
        # Get the four corner pixels
        corners = [
            img_np[0, 0],      # top-left
            img_np[0, -1],     # top-right
            img_np[-1, 0],     # bottom-left
            img_np[-1, -1],    # bottom-right
        ]
        
        # Check if any corner is within the tolerance of the target threshold
        # We use a strict tolerance (default 2) because the padding is usually exact
        return any(
            all(abs(int(c) - threshold) <= tolerance for c in corner)
            for corner in corners
        )
    except Exception as e:
        print(f"[error] Could not process {image_path}: {e}")
        return False

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze padding proportions in training set.")
    parser.add_argument("--annotations", default="coco_dataset/contextual_crops/annotations/single_instances_train.json")
    parser.add_argument("--images-dir", default="coco_dataset/contextual_crops/images/train")
    parser.add_argument("--target-classes", nargs="+", default=["toaster", "hair drier"])
    args = parser.parse_args()

    ann_path = Path(args.annotations)
    images_dir = Path(args.images_dir)

    if not ann_path.exists():
        print(f"[error] Annotations not found: {ann_path}")
        return

    print(f"[init] Loading annotations from {ann_path}...")
    with ann_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Map class names to IDs
    cat_lookup = {c["name"]: c["id"] for c in data.get("categories", [])}
    target_ids = {cat_lookup[name] for name in args.target_classes if name in cat_lookup}
    
    if not target_ids:
        print(f"[error] None of the target classes {args.target_classes} found in categories.")
        return

    # 2. Map image IDs to file names
    img_lookup = {img["id"]: img["file_name"] for img in data.get("images", [])}

    # 3. Identify image paths for target classes
    class_images = {name: [] for name in args.target_classes}
    for ann in data.get("annotations", []):
        cat_id = ann["category_id"]
        for name in args.target_classes:
            if cat_lookup.get(name) == cat_id:
                img_id = ann["image_id"]
                if img_id in img_lookup:
                    class_images[name].append(img_lookup[img_id])
                break

    # 4. Analyze padding for each class
    print("\n--- Padding Analysis (Target Classes) ---")
    total_found = 0
    total_padded = 0

    for name, file_names in class_images.items():
        if not file_names:
            print(f"{name:12}: No images found.")
            continue
        
        padded_count = 0
        valid_file_names = []
        
        # Filter to ensure files exist on disk
        for fname in file_names:
            p = images_dir / fname
            if p.exists():
                valid_file_names.append(p)
        
        if not valid_file_names:
            print(f"{name:12}: Found in JSON, but 0 files exist in {images_dir}.")
            continue

        print(f"Processing {name} ({len(valid_file_names)} images)...")
        for p in tqdm(valid_file_names, leave=False):
            if check_padding(p):
                padded_count += 1
        
        proportion = padded_count / len(valid_file_names)
        print(f"{name:12}: {padded_count}/{len(valid_file_names)} padded ({proportion:.1%})")
        
        total_found += len(valid_file_names)
        total_padded += padded_count

    if total_found > 0:
        overall_prop = total_padded / total_found
        print("-" * 40)
        print(f"{'OVERALL':12}: {total_padded}/{total_found} padded ({overall_prop:.1%})")
    else:
        print("\n[error] No images were processed. Check your paths.")

if __name__ == "__main__":
    main()
