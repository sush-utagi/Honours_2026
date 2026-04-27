#!/usr/bin/env python3
"""Plot class distributions for contextual crops (train/val/test).

Reads the single-instance COCO-format annotations under
coco_dataset/contextual_crops/annotations and shows a bar chart per split.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # safe default; switch to interactive if desired
import matplotlib.pyplot as plt

CONTEXT_ROOT = Path(__file__).resolve().parents[2] / "coco_dataset" / "contextual_crops"
ANN_DIR = CONTEXT_ROOT / "annotations"
SPLITS = ["train", "val", "test"]
# SPLITS = ["train"]


def load_counts(split: str) -> tuple[List[str], Counter]:
    ann_path = ANN_DIR / f"single_instances_{split}.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Missing annotations for split '{split}': {ann_path}")

    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cat_id_to_name: Dict[int, str] = {c["id"]: c["name"] for c in data.get("categories", [])}
    counts = Counter()
    for ann in data.get("annotations", []):
        cname = cat_id_to_name.get(ann["category_id"], f"id_{ann['category_id']}")
        counts[cname] += 1
    ordered_names = [cat_id_to_name[c["id"]] for c in sorted(data.get("categories", []), key=lambda c: c["id"])]
    return ordered_names, counts


def plot_split(split: str, names: List[str], counts: Counter, ax, is_bottom: bool):
    sorted_names = sorted(names, key=lambda n: counts.get(n, 0), reverse=True)
    values = [counts.get(n, 0) for n in sorted_names]
    
    ax.bar(range(len(sorted_names)), values, color="#4C8BF5", edgecolor='black', linewidth=0.5)
    ax.set_title(f"{split.capitalize()} Split ({sum(values):,} images)", fontsize=28, fontweight="bold", pad=25)
    ax.set_ylabel("Count", fontsize=18, fontweight="bold")
    
    ax.set_xticks(range(len(sorted_names)))
    if is_bottom:
        ax.set_xticklabels(sorted_names, rotation=90, fontsize=14)
    else:
        ax.set_xticklabels([])

    ax.grid(axis='y', linestyle='--', alpha=0.7)


def main():
    num_splits = len(SPLITS)
    fig, axes = plt.subplots(num_splits, 1, figsize=(16, 6 * num_splits), sharex=True, constrained_layout=True)
    
    if num_splits == 1:
        axes_list = [axes]
    else:
        axes_list = axes.flatten()

    for i, split in enumerate(SPLITS):
        names, counts = load_counts(split)
        is_bottom = (i == (num_splits - 1))
        plot_split(split, names, counts, axes_list[i], is_bottom)
    fig_dir = Path(__file__).resolve().parents[2] / "experiments" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{'_'.join(SPLITS)}_class_distribution.png"
    out_path = fig_dir / fname
    plt.savefig(out_path, dpi=200)
    print(f"[done] Saved class distribution plot to {out_path}")


if __name__ == "__main__":
    main()
