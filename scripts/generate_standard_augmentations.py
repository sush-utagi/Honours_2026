"""Generate offline standard augmentations for specific minority classes.

Usage:
    python scripts/generate_standard_augmentations.py
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image
import torchvision.transforms as T

def load_annotations(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate standard augmentations.")
    parser.add_argument("--annotations", default="coco_dataset/contextual_crops/annotations/single_instances_train.json")
    parser.add_argument("--images-dir", default="coco_dataset/contextual_crops/images/train")
    parser.add_argument("--output-dir", default="data_generation_outputs/standard_augmentations")
    parser.add_argument("--target-classes", nargs="+", default=["toaster", "hair drier"])
    parser.add_argument("--count", type=int, default=700)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    
    ann_path = Path(args.annotations)
    images_dir = Path(args.images_dir)
    out_dir = Path(args.output_dir)

    print(f"[init] Loading annotations from {ann_path}...")
    data = load_annotations(ann_path)
    
    # Map class names to IDs
    cat_lookup = {c["name"]: c["id"] for c in data.get("categories", [])}
    
    # Map image IDs to file names
    img_lookup = {img["id"]: img["file_name"] for img in data.get("images", [])}

    # Group annotations by category
    class_to_images = {name: [] for name in args.target_classes}
    for ann in data.get("annotations", []):
        cat_id = ann["category_id"]
        for name in args.target_classes:
            if name in cat_lookup and cat_lookup[name] == cat_id:
                img_id = ann["image_id"]
                if img_id in img_lookup:
                    class_to_images[name].append(img_lookup[img_id])
                break

    # Define standard augmentation pipeline
    # Apply ColorJitter BEFORE rotation to ensure the padded areas (fill) stay clean.
    # Increased rotation to 45 degrees and use 127 for neutral padding.
    augment = T.Compose([
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=45, fill=(127, 127, 127)),
    ])

    total_generated = 0
    for name in args.target_classes:
        images = class_to_images.get(name, [])
        if not images:
            print(f"[skip] No images found for class '{name}'.")
            continue
            
        print(f"[process] Class '{name}': found {len(images)} original images. Generating {args.count} augmentations...")
        
        class_out_dir = out_dir / name
        class_out_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(args.count):
            src_img_name = random.choice(images)
            src_path = images_dir / src_img_name
            
            with Image.open(src_path) as img:
                img = img.convert("RGB")
                aug_img = augment(img)
                
                out_name = f"{name}_aug_{i:04d}.jpg"
                aug_img.save(class_out_dir / out_name, format="JPEG", quality=95)
                
        total_generated += args.count
        print(f"[ok] Class '{name}': saved {args.count} augmented images to {class_out_dir}")

    print(f"[done] Generated {total_generated} images in total.")

if __name__ == "__main__":
    main()