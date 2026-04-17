#!/usr/bin/env python3
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

TARGET_CLASSES: dict[str, str] = {
    "toaster": "toaster",
    "hair drier": "hairdryer",
}

CONTROLNET_ANNOTATIONS = "../../coco_dataset/contextual_crops/annotations/single_instances_train.json"
CONTROLNET_IMAGES_DIR = "../../coco_dataset/contextual_crops/images/train"


def score_edge_map(
    image_path: str,
    low_threshold: int = 100,
    high_threshold: int = 200,
) -> tuple[float, float] | tuple[None, None]:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    img_resized = cv2.resize(img, (512, 512))
    edges = cv2.Canny(img_resized, low_threshold, high_threshold)
    variance = float(np.var(edges))
    density = float(np.mean(edges > 0))
    return variance, density


def is_usable_conditioning_image(
    image_path: str,
    min_variance: float = 100.0,
    min_density: float = 0.01,
    max_density: float = 0.50,
) -> bool:
    variance, density = score_edge_map(image_path)
    if variance is None:
        return False
    return variance >= min_variance and min_density <= density <= max_density


def build_controlnet_candidates(
    classes: dict[str, str],
    annotations_path: str = CONTROLNET_ANNOTATIONS,
    images_dir: str = CONTROLNET_IMAGES_DIR,
) -> dict[str, list[str]]:
    script_dir = Path(__file__).resolve().parent
    ann_path = (script_dir / annotations_path).resolve()
    img_dir = (script_dir / images_dir).resolve()

    if not ann_path.exists():
        print(f"[error] Annotations not found: {ann_path}")
        return {cls: [] for cls in classes}

    print(f"[info] Loading annotations from {ann_path} ...")
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    id_to_filename: dict[int, str] = {img["id"]: img["file_name"] for img in coco["images"]}
    id_to_category: dict[int, str] = {cat["id"]: cat["name"] for cat in coco["categories"]}

    category_to_image_ids: dict[str, set[int]] = {}
    for ann in coco["annotations"]:
        cat_name = id_to_category.get(ann["category_id"])
        if cat_name is None:
            continue
        category_to_image_ids.setdefault(cat_name, set()).add(ann["image_id"])

    candidates: dict[str, list[str]] = {}
    for cls in classes:
        image_ids = category_to_image_ids.get(cls, set())
        class_paths = [
            img_dir / id_to_filename[img_id]
            for img_id in image_ids
            if img_id in id_to_filename and (img_dir / id_to_filename[img_id]).exists()
        ]

        # Filter and capture variance for sorting
        passing_with_scores = []
        for p in tqdm(class_paths, desc=f"Filtering {cls}", unit="img"):
            path_str = str(p.resolve())
            var, dens = score_edge_map(path_str)
            
            # Use the same thresholds as is_usable_conditioning_image
            if var is not None and var >= 100.0 and 0.01 <= dens <= 0.50:
                passing_with_scores.append((path_str, var))

        # Sort by variance descending (more detailed edges first)
        passing_with_scores.sort(key=lambda x: x[1], reverse=True)
        passing = [x[0] for x in passing_with_scores]

        print(f"[info] {cls}: {len(passing)}/{len(class_paths)} crops passed edge map filter.")
        candidates[cls] = passing

    output_dir = script_dir / "controlnet_candidates"
    output_dir.mkdir(parents=True, exist_ok=True)
    for cls, paths in candidates.items():
        safe_cls = cls.replace(" ", "_")
        txt_path = output_dir / f"{safe_cls}_candidates.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            for p in paths:
                f.write(p + "\n")
        print(f"[info] Saved {len(paths)} {cls} candidates to {txt_path}")

    return candidates


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-filter ControlNet conditioning candidates from MS COCO.")
    parser.add_argument("--ann", type=str, default=CONTROLNET_ANNOTATIONS, help="Path to COCO annotations.")
    parser.add_argument("--img", type=str, default=CONTROLNET_IMAGES_DIR, help="Path to COCO images.")
    args = parser.parse_args()

    build_controlnet_candidates(TARGET_CLASSES, annotations_path=args.ann, images_dir=args.img)
    print("\nDONE.")
