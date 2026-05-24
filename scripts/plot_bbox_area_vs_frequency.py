#!/usr/bin/env python3
"""
Plots the long-tail distribution of COCO datasets alongside the mean 
original bounding box area (in square pixels) for each class. 
This empirically demonstrates the "compounding scarcity" problem: 
rare classes not only have fewer samples but also exhibit physically 
smaller/lower-resolution bounding boxes before being cropped and padded.

Usage:
    python scripts/plot_bbox_area_vs_frequency.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# We use the original annotations (before contextual crop resizing) to get true resolution
TRAIN_ANN = PROJECT_ROOT / "coco_dataset" / "split" / "annotations" / "instances_train.json"
OUT_PATH = PROJECT_ROOT / "experiments" / "figures" / "bbox_area_vs_frequency.png"

def main():
    if not TRAIN_ANN.exists():
        print(f"Error: {TRAIN_ANN} not found.")
        return

    print("Loading original annotations...")
    with open(TRAIN_ANN, "r") as f:
        data = json.load(f)

    categories = {c["id"]: c["name"] for c in data.get("categories", [])}
    
    # Track frequencies and areas
    class_areas = {cat_id: [] for cat_id in categories.keys()}
    
    for ann in tqdm(data.get("annotations", []), desc="Processing annotations"):
        cat_id = ann["category_id"]
        # Skip crowd annotations
        if ann.get("iscrowd", 0) == 1:
            continue
            
        bbox = ann.get("bbox", [0, 0, 0, 0])
        # bbox format is [x, y, width, height]
        width, height = bbox[2], bbox[3]
        if width <= 0 or height <= 0:
            continue
            
        area = float(width * height)
        class_areas[cat_id].append(area)

    frequencies = {cat_id: len(areas) for cat_id, areas in class_areas.items()}
    
    # Mean area per class
    mean_areas = {}
    for cat_id, areas in class_areas.items():
        if areas:
            mean_areas[cat_id] = np.mean(areas)
        else:
            mean_areas[cat_id] = 0.0

    # Sort categories descending by frequency
    sorted_cats = sorted(categories.keys(), key=lambda c: frequencies[c], reverse=True)
    
    # Remove classes with 0 frequency (just in case some are empty)
    sorted_cats = [c for c in sorted_cats if frequencies[c] > 0]

    # Plotting
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    x_labels = [categories[c] for c in sorted_cats]
    freq_data = [frequencies[c] for c in sorted_cats]
    area_data = [mean_areas[c] for c in sorted_cats]

    fig, ax1 = plt.subplots(figsize=(18, 8))

    # Bar chart for Frequency (Left Y-Axis)
    color1 = '#3b82f6'  # Nice modern blue
    ax1.set_xlabel('COCO Categories (ordered by frequency)', fontsize=14, labelpad=15)
    ax1.set_ylabel('Number of Training Samples', color=color1, fontsize=14, labelpad=10)
    bars = ax1.bar(range(len(x_labels)), freq_data, color=color1, alpha=0.7, label='Class Frequency')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    ax1.set_xticks(range(len(x_labels)))
    ax1.set_xticklabels(x_labels, rotation=90, fontsize=9)
    ax1.set_xlim(-1, len(x_labels))

    # Line/Scatter for Mean BBox Area (Right Y-Axis)
    ax2 = ax1.twinx()  
    color2 = '#10b981'  # Nice modern green
    ax2.set_ylabel('Mean Bounding Box Area (Square Pixels)', color=color2, fontsize=14, labelpad=10)  
    scatter = ax2.plot(range(len(x_labels)), area_data, 'o-', color=color2, markersize=4, linewidth=1.5, label='Mean BBox Area')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Fit a linear trendline to emphasize the degradation
    z = np.polyfit(range(len(area_data)), area_data, 1)
    p = np.poly1d(z)
    trendline = ax2.plot(range(len(x_labels)), p(range(len(area_data))), "--", color='#111827', linewidth=2, alpha=0.8, label="Area Trendline")

    fig.tight_layout()
    plt.title("Compounding Scarcity: Class Frequency vs. Original Bounding Box Area", fontsize=18, fontweight='bold', pad=20)
    
    handles = [bars, scatter[0], trendline[0]]
    labels = [h.get_label() for h in handles]
    ax1.legend(handles, labels, loc='upper right', fontsize=12, framealpha=0.9)

    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.savefig(OUT_PATH, dpi=300, bbox_inches='tight')
    print(f"\n[plot] Success! Saved BBox Area vs Frequency plot to: {OUT_PATH}")

if __name__ == "__main__":
    main()
