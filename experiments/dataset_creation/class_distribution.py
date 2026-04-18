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

    # Ensure consistent ordering by category id/name
    ordered_names = [cat_id_to_name[c["id"]] for c in sorted(data.get("categories", []), key=lambda c: c["id"])]
    return ordered_names, counts


def plot_split(split: str, names: List[str], counts: Counter, ax):
    # Sort classes from highest count to lowest
    sorted_names = sorted(names, key=lambda n: counts.get(n, 0), reverse=True)
    values = [counts.get(n, 0) for n in sorted_names]
    ax.bar(range(len(sorted_names)), values, color="#4C8BF5")
    ax.set_title(f"{split} (n={sum(values):,})")
    ax.set_ylabel("Count")
    ax.set_xticks(range(len(sorted_names)))
    ax.set_xticklabels(sorted_names, rotation=90, fontsize=7)


def main():
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), constrained_layout=True)
    axes = axes.flatten()
    for i, split in enumerate(SPLITS):
        names, counts = load_counts(split)
        plot_split(split, names, counts, axes[i])

    out_path = CONTEXT_ROOT / "class_distribution.png"
    plt.savefig(out_path, dpi=200)
    print(f"[done] Saved class distribution plot to {out_path}")


if __name__ == "__main__":
    main()
