#!/usr/bin/env python3
import json
import random
import argparse
from pathlib import Path
# from parts_complex import (
#     DEFINING_FEATURES, MATERIALS, CONTEXTS, STYLES,
#     QUALITY_MODIFIERS, FRAMING_MODIFIERS, NEGATIVE_PROMPTS
# )

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

TARGET_CLASSES: dict[str, str] = {
    "toaster": "<coco-toaster>",
    "hair drier": "<coco-dryer>",
}

EMBEDDING_PATHS: dict[str, str] = {
    "toaster": "data_generation_backend/embeddings/toaster/learned_embeds.safetensors",
    "hair drier": "data_generation_backend/embeddings/dryer/learned_embeds.safetensors",
}

INCLUDE_SEGMENTATIONS = False
INIT_IMG_PROBABILITY: dict[str, float] = {cls: 1.0 for cls in TARGET_CLASSES}


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

def load_coco_index(annotations_path: Path) -> dict[str, list[tuple[str, list]]]:
    """
    Parse COCO instances and segmentations JSONs, returning a mapping of
    category_name -> [list of (absolute_image_path, segmentation_mask)].
    Only paths that actually exist on disk are included.
    Built once and reused across all classes.
    """
    with open(annotations_path, "r", encoding="utf-8") as f: coco = json.load(f)

    seg_path = annotations_path.parent / annotations_path.name.replace("instances", "segmentations")
    if not seg_path.exists():
        print(f"[warn] Segmentations not found at {seg_path}")
        coco_seg = {"annotations": []}
    else:
        with open(seg_path, "r", encoding="utf-8") as f:
            coco_seg = json.load(f)

    images_dir = annotations_path.parent.parent / "images" / "train"

    # id -> filename
    id_to_filename: dict[int, str] = { img["id"]: img["file_name"] for img in coco["images"] }

    # category_id -> category_name
    id_to_category: dict[int, str] = { cat["id"]: cat["name"] for cat in coco["categories"] }

    # category_name -> set of image_ids that contain at least one instance
    category_to_image_ids: dict[str, set[int]] = {}
    for ann in coco["annotations"]:
        cat_name = id_to_category.get(ann["category_id"])
        if cat_name is None: continue
        category_to_image_ids.setdefault(cat_name, set()).add(ann["image_id"])

    # image_id -> segmentation mask
    id_to_segmentation: dict[int, list] = { ann["image_id"]: ann["segmentation"] for ann in coco_seg.get("annotations", [])}

    # Resolve image_ids to absolute paths and segmentations, skipping missing files
    index: dict[str, list[tuple[str, list]]] = {}
    for cat_name, image_ids in category_to_image_ids.items():
        items = []
        for img_id in image_ids:
            filename = id_to_filename.get(img_id)
            if filename is None:
                continue
            full_path = images_dir / filename
            if full_path.exists():
                seg = id_to_segmentation.get(img_id, [])
                items.append((str(full_path.resolve()), seg))
        if items:
            index[cat_name] = items

    return index


def pick_training_image(cls: str, coco_index: dict[str, list[tuple[str, list]]]) -> tuple[str, list]:
    """
    Returns a random (training image path, segmentation) for the given class,
    or ("", []) if none are available.
    """
    candidates = coco_index.get(cls, [])
    if not candidates:
        print(f"Warning: no training images found for class '{cls}', falling back to txt2img.")
        return "", []
    return random.choice(candidates)


def generate_and_save_class_jsons(classes: dict[str, str], num_per_class: int = 100) -> None:
    cwd = Path.cwd()

    annotations_path = Path("../../coco_dataset/contextual_crops/annotations/single_instances_train.json")

    # Load and index the COCO annotations once, reuse across all classes
    if annotations_path.exists():
        print(f"Loading COCO annotations from {annotations_path} ...")
        coco_index = load_coco_index(annotations_path)
        print(f"Indexed {sum(len(v) for v in coco_index.values())} images across {len(coco_index)} categories.")
        use_img2img = True
    else:
        print(f"Warning: annotations not found at {annotations_path}. All samples will be txt2img.")
        coco_index = {}
        use_img2img = False

    for cls, placeholder in classes.items():
        p_img2img = INIT_IMG_PROBABILITY.get(cls, 0.5)

        samples = []
        for _ in range(num_per_class):
            init_image, segmentation = "", []
            if use_img2img and random.random() < p_img2img:
                init_image, segmentation = pick_training_image(cls, coco_index)

            samples.append(
                {
                    "prompt": build_prompt(cls, placeholder),
                    "negative_prompt": NEGATIVE_PROMPTS.get(cls, _BASE_NEG),
                    "cfg_scale": round(random.uniform(8.8, 9.2), 1),
                    "init_image": init_image,  # "" -> txt2img, path -> img2img
                    "segmentation": segmentation if INCLUDE_SEGMENTATIONS else [],
                }
            )

        final_data = {
            "coco_class": cls,
            "embedding_path": EMBEDDING_PATHS.get(cls, ""),
            "samples": samples,
        }

        filename = f"{cls.replace(' ', '_')}_prompts.json"
        with open(cwd / filename, "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=4)

        img2img_count = sum(1 for s in samples if s["init_image"])
        txt2img_count = num_per_class - img2img_count
        print(f"[done] Saved {filename} with {num_per_class} samples ({img2img_count} img2img @ p={p_img2img}, {txt2img_count} txt2img).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate image prompts for specific COCO classes.")
    parser.add_argument("-n", type=int, default=10, help="Number of prompts to generate per class (default: 10).")
    args = parser.parse_args()
    generate_and_save_class_jsons(TARGET_CLASSES, num_per_class=args.n)
