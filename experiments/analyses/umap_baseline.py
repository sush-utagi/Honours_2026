#!/usr/bin/env python3
"""UMAP feature-space visualisation for the baseline ResNet-18 model.

Compares Real COCO *Train* vs Real COCO *Test* instances for a fixed set of
classes by extracting GAP (Global Average Pooling) features from the trained
checkpoint, reducing them to 2-D with UMAP, and plotting a scatter.

Usage (from project root):
    python experiments/analyses/umap_baseline.py

Optional flags:
    --ckpt          Path to best.pt checkpoint
    --n-per-class   Number of images per class per split  (default 20)
    --save-path     Where to save the figure              (default: experiments/figures/umap_baseline.png)
    --seed          Reproducibility seed                   (default 42)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# ── Ensure project modules are importable ─────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]  # …/Honours_2026
sys.path.insert(0, str(ROOT))

from evaluation_module.classifier.resnet_classifier import build_resnet18

# ── MPS environment hints ─────────────────────────────────────────────────
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

# ── Constants ─────────────────────────────────────────────────────────────
TARGET_CLASSES: List[str] = [
    "person",
    # "book",
    # "toothbrush",
    # "backpack",
    # "parking meter",
    "scissors",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ═══════════════════════════════════════════════════════════════════════════
# Device selection
# ═══════════════════════════════════════════════════════════════════════════

def select_device() -> torch.device:
    """Prefer Apple-Silicon MPS → CUDA → CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ═══════════════════════════════════════════════════════════════════════════
# Model loading + GAP hook
# ═══════════════════════════════════════════════════════════════════════════

class GAPFeatureExtractor:
    """Wraps a ResNet model; intercepts the avgpool output to get 512-D features."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self._features: Optional[torch.Tensor] = None
        # Register a forward hook on the GAP layer
        self.model.avgpool.register_forward_hook(self._hook)

    def _hook(self, _module: torch.nn.Module, _input: Tuple, output: torch.Tensor) -> None:
        # output shape: (B, 512, 1, 1) → squeeze to (B, 512)
        self._features = output.detach().squeeze(-1).squeeze(-1)

    @torch.no_grad()
    def extract(self, batch: torch.Tensor) -> np.ndarray:
        """Run a batch through the model and return the (B, 512) GAP features."""
        self.model(batch)
        assert self._features is not None, "Hook did not fire — check model architecture."
        return self._features.cpu().numpy()


def load_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_resnet18(num_classes=80)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


# ═══════════════════════════════════════════════════════════════════════════
# Annotation-driven image collection
# ═══════════════════════════════════════════════════════════════════════════



def collect_images_for_split(
    contextual_root: Path,
    split: str,
    target_classes: List[str],
    n_per_class: int,
    seed: int,
) -> List[Tuple[Path, str]]:
    """Return up to *n_per_class* (path, class_name) pairs per target class from the specified split."""
    import sys
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from dataset_creation.dataset_assembler import HybridDatasetAssembler
    
    assembler = HybridDatasetAssembler(contextual_root=str(contextual_root), seed=seed)
    paths_by_class = assembler.get_image_paths_by_class(
        split=split,
        target_classes=target_classes,
        max_samples=n_per_class,
    )
    
    result: List[Tuple[Path, str]] = []
    for class_name in target_classes:
        paths = paths_by_class.get(class_name, [])
        if len(paths) == 0:
            print(f"[warn] no images found for class '{class_name}' in split '{split}'")
        elif len(paths) < n_per_class:
            print(f"[warn] only {len(paths)} images available for '{class_name}' in split '{split}' (requested {n_per_class})")
        for p in paths:
            result.append((p, class_name))

    return result


def collect_images_from_directory(
    images_dir: Path,
    target_classes: List[str],
    n_per_class: int,
    seed: int,
) -> List[Tuple[Path, str]]:
    """Walk an ImageFolder-style directory and return up to *n_per_class* per class.

    Expects structure:  images_dir/<class_name>/<image_file>
    Falls back to annotation-based loading if the directory structure is flat.
    """
    result: List[Tuple[Path, str]] = []
    rng = random.Random(seed)

    for class_name in target_classes:
        class_dir = images_dir / class_name
        if not class_dir.is_dir():
            # Try replacing spaces with underscores
            class_dir = images_dir / class_name.replace(" ", "_")
        if not class_dir.is_dir():
            continue

        paths = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        )
        rng.shuffle(paths)
        selected = paths[: min(n_per_class, len(paths))]
        if len(selected) < n_per_class:
            print(
                f"[warn] only {len(selected)} images in '{class_dir}' "
                f"(requested {n_per_class})"
            )
        for p in selected:
            result.append((p, class_name))

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Feature extraction pipeline
# ═══════════════════════════════════════════════════════════════════════════

def build_preprocess() -> transforms.Compose:
    """Standard torchvision preprocessing for ResNet inference."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def extract_features(
    image_items: List[Tuple[Path, str]],
    extractor: GAPFeatureExtractor,
    device: torch.device,
    batch_size: int = 32,
) -> Tuple[np.ndarray, List[str]]:
    """Load images, run through model, and return (N, 512) features + class labels."""
    preprocess = build_preprocess()
    all_features: List[np.ndarray] = []
    all_labels: List[str] = []

    # Process in batches
    for start in tqdm(range(0, len(image_items), batch_size), desc="Extracting features"):
        batch_items = image_items[start : start + batch_size]
        tensors = []
        labels = []
        for path, cls_name in batch_items:
            try:
                img = Image.open(path).convert("RGB")
                tensor = preprocess(img)
                tensors.append(tensor)
                labels.append(cls_name)
            except Exception as exc:
                print(f"[skip] {path}: {exc}")
                continue

        if not tensors:
            continue

        batch = torch.stack(tensors).to(device)
        features = extractor.extract(batch)
        all_features.append(features)
        all_labels.extend(labels)

    return np.concatenate(all_features, axis=0), all_labels


# ═══════════════════════════════════════════════════════════════════════════
# UMAP + Plotting
# ═══════════════════════════════════════════════════════════════════════════

def run_umap(features: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1, seed: int = 42) -> np.ndarray:
    """Project features from 512-D → 2-D using UMAP (on CPU)."""
    try:
        import umap
    except ImportError:
        raise ImportError(
            "The 'umap-learn' package is required.  Install with:\n"
            "  pip install umap-learn"
        )

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric="euclidean",
        random_state=seed,
    )
    return reducer.fit_transform(features)


def plot_umap(
    embedding: np.ndarray,
    labels: List[str],
    splits: List[str],
    classes: List[str],
    save_path: Path,
) -> None:
    """Create a scatter plot with colours by class and markers by split."""

    # Colour palette — one colour per class
    cmap = plt.cm.get_cmap("tab10", len(classes))
    class_to_color = {cls: cmap(i) for i, cls in enumerate(classes)}

    # Marker map
    split_to_marker = {"train": "o", "test": "x"}
    split_to_size   = {"train": 30, "test": 50}

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each combination of (class, split) as a separate scatter call for legend
    for split in ["train", "test"]:
        for cls in classes:
            mask = np.array(
                [(l == cls and s == split) for l, s in zip(labels, splits)],
                dtype=bool,
            )
            if not mask.any():
                continue
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=[class_to_color[cls]],
                marker=split_to_marker[split],
                s=split_to_size[split],
                alpha=0.75,
                edgecolors="none" if split == "train" else "k",
                linewidths=0.4,
                label=f"{cls} ({split})",
            )

    ax.set_title("UMAP Feature Space — Baseline ResNet-18\nTrain (●) vs Test (✕)", fontsize=14, fontweight="bold")
    ax.set_xlabel("UMAP-1", fontsize=11)
    ax.set_ylabel("UMAP-2", fontsize=11)

    # Build a clean legend: grouped by class then split
    handles, leg_labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        leg_labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        framealpha=0.9,
        title="Class (split)",
        title_fontsize=9,
        ncol=1,
    )

    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] figure saved to {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="UMAP feature-space visualisation for baseline ResNet-18",
    )
    parser.add_argument(
        "--ckpt",
        default="runs/baseline_model_A/checkpoints/best.pt",
        help="Path to the best.pt checkpoint.",
    )
    parser.add_argument(
        "--train-ann",
        default="coco_dataset/contextual_crops/annotations/single_instances_train.json",
        help="COCO annotation file for training images.",
    )
    parser.add_argument(
        "--train-img-dir",
        default="coco_dataset/contextual_crops/images/train",
        help="Directory containing training images.",
    )
    parser.add_argument(
        "--test-ann",
        default="coco_dataset/contextual_crops/annotations/single_instances_test.json",
        help="COCO annotation file for test images.",
    )
    parser.add_argument(
        "--test-img-dir",
        default="coco_dataset/contextual_crops/images/test",
        help="Directory containing test images.",
    )
    parser.add_argument(
        "--n-per-class",
        type=int,
        default=20,
        help="Number of images to sample per class per split.",
    )
    parser.add_argument(
        "--save-path",
        default="experiments/figures/umap_baseline.png",
        help="Where to save the output figure.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
    )
    args = parser.parse_args()

    # ── Resolve paths relative to project root ────────────────────────────
    ckpt_path     = ROOT / args.ckpt
    train_ann     = ROOT / args.train_ann
    train_img_dir = ROOT / args.train_img_dir
    test_ann      = ROOT / args.test_ann
    test_img_dir  = ROOT / args.test_img_dir
    save_path     = ROOT / args.save_path

    for p, label in [
        (ckpt_path, "Checkpoint"),
        (train_ann, "Train annotations"),
        (train_img_dir, "Train images dir"),
        (test_ann, "Test annotations"),
        (test_img_dir, "Test images dir"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{label} not found: {p}")

    # ── Device + model ────────────────────────────────────────────────────
    device = select_device()
    print(f"[device] {device}")

    model = load_model(ckpt_path, device)
    extractor = GAPFeatureExtractor(model)

    # ── Collect images ────────────────────────────────────────────────────
    contextual_root = ROOT / "coco_dataset" / "contextual_crops"
    
    print(f"\n[data] collecting {args.n_per_class} images/class from TRAIN …")
    train_items = collect_images_for_split(
        contextual_root, "train", TARGET_CLASSES, args.n_per_class, args.seed,
    )
    print(f"[data] collected {len(train_items)} train images")

    print(f"[data] collecting {args.n_per_class} images/class from TEST …")
    test_items = collect_images_for_split(
        contextual_root, "test", TARGET_CLASSES, args.n_per_class, args.seed + 1,
    )
    print(f"[data] collected {len(test_items)} test images")

    # ── Extract features ──────────────────────────────────────────────────
    print("\n[model] extracting train features …")
    train_feats, train_labels = extract_features(train_items, extractor, device, args.batch_size)
    print(f"[model] train features shape: {train_feats.shape}")

    print("[model] extracting test features …")
    test_feats, test_labels = extract_features(test_items, extractor, device, args.batch_size)
    print(f"[model] test features shape: {test_feats.shape}")

    # ── Combine ───────────────────────────────────────────────────────────
    all_features = np.concatenate([train_feats, test_feats], axis=0)
    all_labels   = train_labels + test_labels
    all_splits   = ["train"] * len(train_labels) + ["test"] * len(test_labels)

    print(f"\n[umap] projecting {all_features.shape[0]} × {all_features.shape[1]} → 2-D …")
    embedding = run_umap(all_features, n_neighbors=15, min_dist=0.1, seed=args.seed)
    print(f"[umap] done — embedding shape: {embedding.shape}")

    # ── Plot ──────────────────────────────────────────────────────────────
    plot_umap(embedding, all_labels, all_splits, TARGET_CLASSES, save_path)


if __name__ == "__main__":
    main()
