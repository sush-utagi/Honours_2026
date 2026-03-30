#!/usr/bin/env python3
"""Train Model A (baseline) on contextual COCO crops without pretraining."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "dataset-creation"))
sys.path.append(str(ROOT / "evaluation-module" / "classifier"))

from dataset_assembler import HybridDatasetAssembler
from resnet_classifier import (
    build_resnet18,
    train_model,
    evaluate,
    plot_training_curves,
    plot_precision_recall_curves,
    compute_gradcam,
)


def select_device() -> torch.device:
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
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
    print(f"[device] using {device}")

    output_dir = ROOT / "runs" / "baseline_model_A"
    output_dir.mkdir(parents=True, exist_ok=True)

    assembler = HybridDatasetAssembler(contextual_root="coco_dataset/contextual_crops")
    datasets = assembler.assemble(synthetic_dir_name=None, target_class_name=None)
    train_ds = datasets.get("train")
    val_ds = datasets.get("val")
    if train_ds is None or val_ds is None:
        raise RuntimeError("Train/val splits missing from assembled dataset.")
    class_names = assembler.idx_to_name

    train_loader = DataLoader(
        train_ds,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_without_synthetic,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_without_synthetic,
    )

    model = build_resnet18(num_classes=80)  # random init; no ImageNet weights
    history = train_model(
        model,
        train_loader,
        val_loader,
        epochs=10,
        lr=1e-4,
        device=device,
        weight_decay=1e-4,
    )

    criterion = torch.nn.CrossEntropyLoss()
    val_loss, val_acc, y_true, y_prob = evaluate(model.to(device), val_loader, criterion, device)
    print(f"[val] loss={val_loss:.4f} acc={val_acc:.3f}")

    plot_training_curves(history, save_path=str(output_dir / "training_curves.png"))
    plot_precision_recall_curves(y_true, y_prob, class_names, save_path=str(output_dir / "precision_recall.png"))

    gradcam_dir = output_dir / "gradcam"
    generate_gradcam_overlays(model, val_loader, class_names, gradcam_dir, device=device, max_images=6)


if __name__ == "__main__":
    main()
