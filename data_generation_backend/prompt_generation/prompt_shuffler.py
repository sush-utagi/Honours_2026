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
    "toaster": "toaster",
    "hair drier": "hairdryer",
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

    # write controlnet candidates to .txt files
    output_dir = Path(__file__).resolve().parent / "controlnet_candidates"
    output_dir.mkdir(parents=True, exist_ok=True)
    for cls, paths in candidates.items():
        safe_cls = cls.replace(" ", "_")
        txt_path = output_dir / f"{safe_cls}_candidates.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            for p in paths:
                f.write(p + "\n")
        print(f"[info] Saved candidate list for {cls} to {txt_path}")

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


def sample_ip_adapter_images(cls: str, num_samples: int) -> list[str]:
    safe_cls = cls.replace(" ", "_")
    script_dir = Path(__file__).resolve().parent
    meta_path = script_dir / "selected_references" / safe_cls / "metadata.json"
    
    if not meta_path.exists():
        print(f"[warn] No metadata found for IP adapter at {meta_path}.")
        return [""] * num_samples
        
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    refs = data.get("references", [])
    if not refs:
        print(f"[warn] No references array found in {meta_path}.")
        return [""] * num_samples
        
    paths = [r["original_path"] for r in refs]
    weights = [r["cluster_size"] for r in refs]
    
    total = sum(weights)
    if total == 0:
        probs = [1.0 / len(weights)] * len(weights)
    else:
        probs = [w / total for w in weights]
        
    chosen = random.choices(paths, weights=probs, k=num_samples)
    return chosen

def generate_and_save_class_jsons(
    classes: dict[str, str],
    num_per_class: int = 100,
    mode: str | None = None,
) -> None:
    cwd = Path.cwd()
    
    modes_to_run = [mode] if mode else ["ip_adapter", "controlnet"]

    all_cn_candidates = {}
    if any(m in ("controlnet", "hybrid") for m in modes_to_run):
        all_cn_candidates = build_controlnet_candidates(classes)

    all_ip_references = {}
    if any(m in ("ip_adapter", "hybrid") for m in modes_to_run):
        for cls in classes:
            all_ip_references[cls] = sample_ip_adapter_images(cls, num_per_class)

    for cls, target_value in classes.items():
        print(f"\n[batch] Generating consistency-locked prompts for: {cls} (using token: {target_value})")
        prompt_token = target_value
        
        base_samples = []
        for _ in range(num_per_class):
            base_samples.append({
                "prompt": build_prompt(cls, prompt_token),
                "negative_prompt": NEGATIVE_PROMPTS.get(cls, _BASE_NEG),
                "cfg_scale": round(random.uniform(5.0, 9.0), 1),
            })

        for current_mode in modes_to_run:
            samples = []
            cn_valid_count = 0
            for i, base in enumerate(base_samples):
                sample = base.copy()
                if current_mode in ("controlnet", "hybrid"):
                    cls_cands = all_cn_candidates.get(cls, [])
                    if cls_cands:
                        sample["controlnet_image"] = random.choice(cls_cands)
                        cn_valid_count += 1
                    else:
                        sample["controlnet_image"] = ""
                
                if current_mode in ("ip_adapter", "hybrid"):
                    sample["ip_image"] = all_ip_references[cls][i]
                
                samples.append(sample)

            final_data = {
                "coco_class": cls,
                "generation_mode": current_mode,
                "samples": samples,
            }

            filename = f"{cls.replace(' ', '_')}_{current_mode}_prompts.json"
            with open(cwd / filename, "w", encoding="utf-8") as f:
                json.dump(final_data, f, indent=4)

            log_msg = f"[done] Saved {filename} ({num_per_class} samples)"
            if current_mode in ("controlnet", "hybrid"):
                log_msg += f" [{cn_valid_count} with source images]"
            print(log_msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate image prompts for specific COCO classes.")
    parser.add_argument("-n", type=int, default=10, help="Number of prompts to generate per class (default: 10).")
    parser.add_argument("--mode", choices=["ip_adapter", "controlnet", "hybrid"], default=None, 
                        help="Generation mode. Default=None (generates both ip_adapter and controlnet with identical prompts).")
    args = parser.parse_args()
    generate_and_save_class_jsons(TARGET_CLASSES, num_per_class=args.n, mode=args.mode)
