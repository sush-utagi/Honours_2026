#!/usr/bin/env python3
import json
import random
import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from parts_simple import (
    DEFINING_FEATURES,
    MATERIALS,
    CONTEXTS,
    SHOT_TYPES,
    LIGHTING,
    FRAMING,
    QUALITY_TAGS,
    NEGATIVE_PROMPTS,
    _BASE_NEG,
)


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


TARGET_CLASSES: dict[str, str] = {
    "toaster": "<coco-toaster>",
    "hair drier": "<coco-dryer>",
}

EMBEDDING_PATHS: dict[str, str] = {
    "toaster": "data_generation_backend/embeddings/toaster/learned_embeds.safetensors",
    "hair drier": "data_generation_backend/embeddings/dryer/learned_embeds.safetensors",
}

CONTROLNET_ANNOTATIONS = "../../coco_dataset/contextual_crops/annotations/single_instances_train.json"
CONTROLNET_IMAGES_DIR = "../../coco_dataset/contextual_crops/images/train"


def build_controlnet_candidates(
    classes: dict[str, str],
    annotations_path: str = CONTROLNET_ANNOTATIONS,
    images_dir: str = CONTROLNET_IMAGES_DIR,
) -> dict[str, list[str]]:
    ann_path = Path(annotations_path)
    img_dir = Path(images_dir)

    if not ann_path.exists():
        print(f"[warn] Annotations not found: {ann_path}. No ControlNet candidates.")
        return {cls: [] for cls in classes}

    print(f"[info] Loading annotations from {ann_path} ...")
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Build lookups
    id_to_filename: dict[int, str] = {img["id"]: img["file_name"] for img in coco["images"]}
    id_to_category: dict[int, str] = {cat["id"]: cat["name"] for cat in coco["categories"]}

    # category_name -> set of image_ids
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

        passing = [
            str(p.resolve())
            for p in tqdm(class_paths, desc=f"Filtering {cls}", unit="img")
            if is_usable_conditioning_image(str(p.resolve()))
        ]
        print(f"[info] {cls}: {len(passing)}/{len(class_paths)} crops passed edge map filter.")
        candidates[cls] = passing

    return candidates



def build_prompt(cls: str, placeholder: str) -> str:
    feature = random.choice(DEFINING_FEATURES.get(cls) or [cls])
    material = random.choice(MATERIALS.get(cls) or [""])
    context = random.choice(CONTEXTS.get(cls) or [""])
    shot = random.choice(SHOT_TYPES)
    lighting = random.choice(LIGHTING)
    framing = random.choice(FRAMING)
    quality = random.choice(QUALITY_TAGS)

    weight = 1.2
    parts = [
        f"A photo of a ({placeholder}){weight}",
        feature,
        f"made of {material}" if material else "",
        context,
        shot,
        lighting,
        framing,
        quality,
    ]
    return ", ".join(p for p in parts if p)


def generate_and_save_class_jsons(
    classes: dict[str, str],
    num_per_class: int = 100,
    mode: str = "ti",
) -> None:
    cwd = Path.cwd()

    # Build ControlNet candidates once if needed
    candidates: dict[str, list[str]] = {}
    if mode == "controlnet":
        candidates = build_controlnet_candidates(classes)

    for cls, placeholder in classes.items():
        warned_empty = False
        controlnet_valid_count = 0

        # ControlNet uses plain words, TI uses the learned placeholder token
        prompt_token = cls if mode == "controlnet" else placeholder

        samples = []
        for _ in range(num_per_class):
            sample: dict = {
                "prompt": build_prompt(cls, prompt_token),
                "negative_prompt": NEGATIVE_PROMPTS.get(cls, _BASE_NEG),
                "cfg_scale": round(random.uniform(5.0, 9.0), 1),
            }

            if mode == "controlnet":
                cls_candidates = candidates.get(cls, [])
                if cls_candidates:
                    sample["controlnet_image"] = random.choice(cls_candidates)
                    controlnet_valid_count += 1
                else:
                    sample["controlnet_image"] = ""
                    if not warned_empty:
                        print(f"[warn] No valid ControlNet conditioning images for class '{cls}'. All samples will have empty controlnet_image.")
                        warned_empty = True

            samples.append(sample)

        final_data = {
            "coco_class": cls,
            "embedding_path": EMBEDDING_PATHS.get(cls, ""),
            "generation_mode": mode,
            "samples": samples,
        }

        filename = f"{cls.replace(' ', '_')}_{mode}_prompts.json"
        with open(cwd / filename, "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=4)

        if mode == "controlnet":
            empty_count = num_per_class - controlnet_valid_count
            print(f"[done] Saved {filename} with {num_per_class} samples (mode={mode}, {controlnet_valid_count} with conditioning image, {empty_count} empty).")
        else:
            print(f"[done] Saved {filename} with {num_per_class} samples (mode={mode}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate image prompts for specific COCO classes.")
    parser.add_argument("-n", type=int, default=10, help="Number of prompts to generate per class (default: 10).")
    parser.add_argument("--mode", choices=["ti", "controlnet"], default="ti", help="Generation mode: 'ti' for text2img with textual inversion, 'controlnet' for ControlNet canny (default: ti).")
    args = parser.parse_args()
    generate_and_save_class_jsons(TARGET_CLASSES, num_per_class=args.n, mode=args.mode)
