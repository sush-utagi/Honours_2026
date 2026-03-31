#!/usr/bin/env python3
"""Train Model A (baseline) on contextual COCO crops without pretraining."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt

# Encourage MPS to fall back instead of crashing when it runs out of supported ops/memory.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "dataset_creation"))
sys.path.append(str(ROOT / "evaluation_module" / "classifier"))

from dataset_creation.dataset_assembler import HybridDatasetAssembler
from evaluation_module.classifier.resnet_classifier import (
    build_resnet18,
    train_model,
    evaluate,
    plot_training_curves,
    plot_precision_recall_curves,
    compute_gradcam,
)


def select_device() -> torch.device:
    use_mps = os.getenv("USE_MPS", "1") == "1"
    if use_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def collate_without_synthetic(batch):
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return images, labels


def generate_gradcam_overlays(
    model: torch.nn.Module,
    loader: DataLoader,
    class_names: Sequence[str],
    output_dir: Path,
    device: torch.device,
    max_images: int = 6,
) -> List[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.to(device).eval()
    saved: List[str] = []
    processed = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        for i in range(images.size(0)):
            if processed >= max_images:
                return saved
            img = images[i]
            target_cls = int(labels[i].item())
            heatmap = compute_gradcam(model, img.cpu(), target_cls, device=device)

            img_np = img.cpu().permute(1, 2, 0).numpy()
            img_np = ((img_np + 1.0) * 0.5).clip(0, 1)  # [-1,1] -> [0,1]

            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(img_np)
            ax.imshow(heatmap, cmap="jet", alpha=0.4)
            ax.axis("off")
            fname = output_dir / f"gradcam_{processed}_{class_names[target_cls]}.png"
            plt.tight_layout()
            plt.savefig(fname, dpi=200)
            plt.close(fig)

            saved.append(str(fname))
            processed += 1
        if processed >= max_images:
            break
    return saved


def main() -> None:
    device = select_device()
    if device.type == "mps" and os.getenv("USE_MPS", "1") != "1":
        print("[device] MPS available but disabled via USE_MPS=0; using CPU instead.")
    print(f"[device] using {device}")
    if device.type == "mps":
        print(f"[device] mps_available={torch.backends.mps.is_available()} mps_built={torch.backends.mps.is_built()}")
        try:
            major, minor = (int(x) for x in torch.__version__.split(".")[:2])
            if (major, minor) < (2, 2):
                print(f"[warn] torch {torch.__version__} has unstable MPS; upgrade to 2.2+ for better reliability.")
        except Exception:
            pass

    output_dir = ROOT / "runs" / "baseline_model_A"
    output_dir.mkdir(parents=True, exist_ok=True)

    assembler = HybridDatasetAssembler(contextual_root="coco_dataset/contextual_crops")
    datasets = assembler.assemble(synthetic_dir_name=None, target_class_name=None)
    train_ds = datasets.get("train")
    val_ds = datasets.get("val")
    if train_ds is None or val_ds is None:
        raise RuntimeError("Train/val splits missing from assembled dataset.")
    class_names = assembler.idx_to_name

    print(f"[data] train samples: {len(train_ds):,} | val samples: {len(val_ds):,}")

    # Derive loader settings per device.
    if device.type == "mps":
        batch_size = 32
        num_workers = 2
        pin_memory = False
    elif device.type == "cuda":
        batch_size = 64
        num_workers = 8
        pin_memory = True
    else:  # cpu
        batch_size = 32
        num_workers = 4
        pin_memory = False
    persistent = num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        collate_fn=collate_without_synthetic,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        collate_fn=collate_without_synthetic,
    )

    model = build_resnet18(num_classes=80)  # random init; no ImageNet weights
    history = train_model(
        model,
        train_loader,
        val_loader,
        epochs=4,
        lr=1e-4,
        device=device,
        weight_decay=1e-4,
        show_progress=True,
        checkpoint_dir=str(output_dir / "checkpoints"),
        save_every=2, 
    )

    criterion = torch.nn.CrossEntropyLoss()
    val_loss, val_acc, y_true, y_prob = evaluate(
        model.to(device),
        val_loader,
        criterion,
        device,
        progress_desc="Val final",
        show_progress=True,
    )
    print(f"[val] loss={val_loss:.4f} acc={val_acc:.3f}")

    plot_training_curves(history, save_path=str(output_dir / "training_curves.png"))
    plot_precision_recall_curves(y_true, y_prob, class_names, save_path=str(output_dir / "precision_recall.png"))

    gradcam_dir = output_dir / "gradcam"
    generate_gradcam_overlays(model, val_loader, class_names, gradcam_dir, device=device, max_images=6)


if __name__ == "__main__":
    main()
