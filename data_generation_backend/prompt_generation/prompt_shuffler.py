#!/usr/bin/env python3
import json
import random
import argparse
from pathlib import Path

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
    "toaster": "toaster",
    "hair drier": "hairdryer",
}

def load_controlnet_candidates(classes: dict[str, str]) -> dict[str, list[str]]:
    candidates = {}
    script_dir = Path(__file__).resolve().parent
    cand_dir = script_dir / "controlnet_candidates"
    
    if not cand_dir.exists():
        print(f"[warn] ControlNet candidate directory not found at {cand_dir}. Run select_controlnet_candidates.py first.")
        return {cls: [] for cls in classes}

    for cls in classes:
        safe_cls = cls.replace(" ", "_")
        txt_path = cand_dir / f"{safe_cls}_candidates.txt"
        
        if txt_path.exists():
            with open(txt_path, "r", encoding="utf-8") as f:
                paths = [line.strip() for line in f if line.strip()]
            candidates[cls] = paths
            print(f"[info] Loaded {len(paths)} ControlNet candidates for {cls}")
        else:
            print(f"[warn] No candidate file found for {cls} at {txt_path}")
            candidates[cls] = []
            
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
    
    # 0. Determine which modes to create
    modes_to_run = [mode] if mode else ["ip_adapter", "controlnet"]

    # 1. Pre-fetch candidates/references for all needed modes
    all_cn_candidates = {}
    if any(m in ("controlnet", "hybrid") for m in modes_to_run):
        all_cn_candidates = load_controlnet_candidates(classes)

    all_ip_references = {}
    if any(m in ("ip_adapter", "hybrid") for m in modes_to_run):
        for cls in classes:
            all_ip_references[cls] = sample_ip_adapter_images(cls, num_per_class)

    # 2. Process each class
    for cls, target_value in classes.items():
        print(f"\n[batch] Generating consistency-locked prompts for: {cls} (using token: {target_value})")
        prompt_token = target_value
        
        # Lock in the prompts for this class so all JSONs use identical text
        base_samples = []
        for _ in range(num_per_class):
            base_samples.append({
                "prompt": build_prompt(cls, prompt_token),
                "negative_prompt": NEGATIVE_PROMPTS.get(cls, _BASE_NEG),
                "cfg_scale": round(random.uniform(5.0, 9.0), 1),
            })

        # 3. Create a separate JSON for each target mode
        for current_mode in modes_to_run:
            samples = []
            cn_valid_count = 0
            for i, base in enumerate(base_samples):
                sample = base.copy()
                
                # Insert ControlNet source if mode requires it
                if current_mode in ("controlnet", "hybrid"):
                    cls_cands = all_cn_candidates.get(cls, [])
                    if cls_cands:
                        sample["controlnet_image"] = random.choice(cls_cands)
                        cn_valid_count += 1
                    else:
                        sample["controlnet_image"] = ""
                
                # Insert IP-Adapter source if mode requires it
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
                        help="Generation mode. Omit to generate both ip_adapter and controlnet JSONs with identical prompts.")
    args = parser.parse_args()
    generate_and_save_class_jsons(TARGET_CLASSES, num_per_class=args.n, mode=args.mode)
