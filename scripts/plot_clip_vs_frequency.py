#!/usr/bin/env python3
"""
Plots the long-tail distribution of COCO contextual crops alongside the mean 
CLIP score for each class. This empirically demonstrates the "compounding scarcity" 
problem: rare classes not only have fewer samples but also exhibit lower semantic 
quality/representativeness against their domain-level text prompt.

Usage:
    python scripts/plot_clip_vs_frequency.py --max-per-class 100
"""

import json
import torch
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import warnings

# Suppress warnings for MPS fallback if needed
warnings.filterwarnings('ignore', category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CROPS_ROOT = PROJECT_ROOT / "coco_dataset" / "contextual_crops"
TRAIN_ANN = CROPS_ROOT / "annotations" / "single_instances_train.json"
TRAIN_IMG_DIR = CROPS_ROOT / "images" / "train"
OUT_PATH = PROJECT_ROOT / "experiments" / "figures" / "clip_vs_frequency.png"

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def get_article(word):
    """Return 'an' if word starts with a vowel, otherwise 'a'."""
    if not word: 
        return "a"
    return "an" if word[0].lower() in 'aeiou' else "a"

def main():
    parser = argparse.ArgumentParser(description="Evaluate CLIP scores of COCO crops against class prompts.")
    parser.add_argument("--max-per-class", type=int, default=100, 
                        help="Max images to evaluate per class (for speed). Set to 0 to use all.")
    parser.add_argument("--model-id", type=str, default="openai/clip-vit-base-patch32",
                        help="CLIP model to use.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for CLIP inference.")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # 1. Load Annotations
    if not TRAIN_ANN.exists():
        print(f"Error: {TRAIN_ANN} not found. Have you generated contextual crops yet?")
        return

    print("Loading annotations...")
    with open(TRAIN_ANN, "r") as f:
        data = json.load(f)

    categories = {c["id"]: c["name"] for c in data.get("categories", [])}
    img_map = {i["id"]: i["file_name"] for i in data.get("images", [])}
    
    # Group images by category
    class_images = {cat_id: [] for cat_id in categories.keys()}
    for ann in data.get("annotations", []):
        cat_id = ann["category_id"]
        img_id = ann["image_id"]
        file_name = img_map.get(img_id)
        if file_name:
            class_images[cat_id].append(file_name)

    # Compute true frequencies
    frequencies = {cat_id: len(imgs) for cat_id, imgs in class_images.items()}

    # Sample for performance if needed
    if args.max_per_class > 0:
        for cat_id in class_images:
            if len(class_images[cat_id]) > args.max_per_class:
                random.seed(42) # Deterministic sampling
                class_images[cat_id] = random.sample(class_images[cat_id], args.max_per_class)

    # 2. Load CLIP Model
    print(f"Loading {args.model_id}...")
    model = CLIPModel.from_pretrained(args.model_id).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_id)
    model.eval()

    mean_clip_scores = {}
    
    # Sort categories descending by frequency
    sorted_cats = sorted(categories.keys(), key=lambda c: frequencies[c], reverse=True)

    print("Evaluating CLIP scores...")
    with torch.no_grad():
        for cat_id in tqdm(sorted_cats, desc="Classes"):
            cat_name = categories[cat_id]
            article = get_article(cat_name)
            
            # Domain-level prompt (accounting for vowels)
            prompt = f"a photo of {article} {cat_name}"
            
            # Prepare text inputs
            text_inputs = processor(text=[prompt], return_tensors="pt", padding=True).to(device)
            text_features = model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            
            scores = []
            files = class_images[cat_id]
            
            # Batch process images
            for i in range(0, len(files), args.batch_size):
                batch_files = files[i:i+args.batch_size]
                images = []
                
                for f in batch_files:
                    path = TRAIN_IMG_DIR / f
                    if path.exists():
                        try:
                            img = Image.open(path).convert("RGB")
                            images.append(img)
                        except Exception:
                            pass
                
                if not images:
                    continue
                    
                image_inputs = processor(images=images, return_tensors="pt").to(device)
                image_features = model.get_image_features(**image_inputs)
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                
                # Cosine similarity
                sim = (image_features @ text_features.T).squeeze(-1)
                scores.extend(sim.cpu().tolist())
                
            if scores:
                mean_clip_scores[cat_id] = np.mean(scores)
            else:
                mean_clip_scores[cat_id] = 0.0

    # 3. Plotting
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    x_labels = [categories[c] for c in sorted_cats]
    freq_data = [frequencies[c] for c in sorted_cats]
    clip_data = [mean_clip_scores.get(c, 0.0) for c in sorted_cats]

    # Use a wide layout to accommodate 80 classes
    fig, ax1 = plt.subplots(figsize=(18, 8))

    # Bar chart for Frequency (Left Y-Axis)
    color1 = '#3b82f6'  # Nice modern blue
    ax1.set_xlabel('COCO Categories (ordered by frequency)', fontsize=14, labelpad=15)
    ax1.set_ylabel('Number of Training Samples', color=color1, fontsize=14, labelpad=10)
    bars = ax1.bar(range(len(x_labels)), freq_data, color=color1, alpha=0.7, label='Class Frequency')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Configure X-axis labels
    ax1.set_xticks(range(len(x_labels)))
    ax1.set_xticklabels(x_labels, rotation=90, fontsize=9)
    ax1.set_xlim(-1, len(x_labels))

    # Line/Scatter for CLIP Score (Right Y-Axis)
    ax2 = ax1.twinx()  
    color2 = '#ef4444'  # Nice modern red
    ax2.set_ylabel('Mean CLIP Score (Cosine Similarity)', color=color2, fontsize=14, labelpad=10)  
    scatter = ax2.plot(range(len(x_labels)), clip_data, 'o-', color=color2, markersize=4, linewidth=1.5, label='Mean CLIP Score')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Fit a linear trendline for the CLIP score to emphasize the degradation
    z = np.polyfit(range(len(clip_data)), clip_data, 1)
    p = np.poly1d(z)
    trendline = ax2.plot(range(len(x_labels)), p(range(len(clip_data))), "--", color='#111827', linewidth=2, alpha=0.8, label="CLIP Trendline")

    fig.tight_layout()
    plt.title("Compounding Scarcity: Class Frequency vs. Semantic Representativeness", fontsize=18, fontweight='bold', pad=20)
    
    # Combine legends into one box
    handles = [bars, scatter[0], trendline[0]]
    labels = [h.get_label() for h in handles]
    ax1.legend(handles, labels, loc='upper right', fontsize=12, framealpha=0.9)

    # Optional: add a grid just for the CLIP scores
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.savefig(OUT_PATH, dpi=300, bbox_inches='tight')
    print(f"\n[plot] Success! Saved CLIP vs Frequency plot to: {OUT_PATH}")

if __name__ == "__main__":
    main()
