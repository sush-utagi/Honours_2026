#!/usr/bin/env python3
"""Train a ResNet18 classifier on contextual COCO crops (no pretraining).

Supports both the baseline model and an experimental variant.

RESUME TRAINING (examples):
    python experiments/model/train_model.py --model-type baseline --epochs 2 --resume-ckpt runs/baseline_model_A/checkpoints/last.pt
    python experiments/model/train_model.py --model-type experimental --epochs 2 --resume-ckpt runs/experimental_model_A/checkpoints/last.pt

NEW TRAINING:
    python experiments/model/train_model.py --model-type baseline --epochs <NUM_EPOCHS>
    python experiments/model/train_model.py --model-type experimental --epochs <NUM_EPOCHS>
"""

from __future__ import annotations

import argparse
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


def resolve_project_path(path_like: str | Path) -> Path:
    """Return an absolute path, resolving relative ones from the project root."""

    path = Path(path_like)
    return path if path.is_absolute() else PROJECT_ROOT / path


from dataset_creation.dataset_assembler import HybridDatasetAssembler
from experiments.evaluation_module.classifier.resnet_classifier import (
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
    parser = argparse.ArgumentParser(description="Train (baseline or experimental) ResNet18 classifier.")
    parser.add_argument(
        "--model-type",
        choices=["baseline", "baseline_B", "experimental"],
        default="baseline",
        help="Model variant; controls run/log directories (default: baseline).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of additional epochs to train (default: 2).",
    )
    parser.add_argument(
        "--resume-ckpt",
        default=None,
        help="Optional checkpoint path to resume training from.",
    )
    args = parser.parse_args()

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

    if args.model_type == "baseline":
        run_name = "baseline_model_A"
        log_name = "log_baseline_A.txt"
    elif args.model_type == "baseline_B":
        run_name = "baseline_model_B"
        log_name = "log_baseline_B.txt"
    else:
        run_name = "experimental_model_A"
        log_name = "log_experimental_A.txt"
    output_dir = PROJECT_ROOT / "runs" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] model_type={args.model_type} run_dir={output_dir}")

    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    log_path = logs_dir / log_name

    assembler = HybridDatasetAssembler(
        contextual_root=str(PROJECT_ROOT / "coco_dataset" / "contextual_crops")
    )
    datasets = assembler.assemble(synthetic_dir_name=None, target_class_name=None, load_test=False)
    train_ds = datasets.get("train")
    val_ds = datasets.get("val")
    if train_ds is None or val_ds is None:
        raise RuntimeError("Train/val splits missing from assembled dataset.")
    class_names = assembler.idx_to_name

    print(f"[data] train samples: {len(train_ds):,} | val samples: {len(val_ds):,}")

    # Derive loader settings per device.
    if device.type == "mps":
        batch_size = 32
        num_workers = 4
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

    start_epoch = 0
    optimizer_state = None
    best_val_acc_init = None
    if args.resume_ckpt:
        ckpt_path = resolve_project_path(args.resume_ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        if "model_state_dict" not in ckpt:
            raise ValueError(f"Checkpoint missing model_state_dict: {ckpt_path}")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer_state = ckpt.get("optimizer_state_dict")
        best_val_acc_init = ckpt.get("best_val_acc")
        start_epoch = int(ckpt.get("epoch", 0))
        print(f"[resume] loaded {ckpt_path} (epoch={start_epoch}, best_val_acc={best_val_acc_init})")

    mode = "a" if args.resume_ckpt else "w"
    with log_path.open(mode, encoding="utf-8") as log_f:
        def log(msg: str) -> None:
            log_f.write(msg + "\n")
            log_f.flush()

        if args.resume_ckpt:
            log("\n" + "=" * 60)
            log("Resumed training run")
        log(f"[run] model_type={args.model_type} run_dir={output_dir}")
        log(f"[device] using {device}")
        log(f"[data] train samples: {len(train_ds):,} | val samples: {len(val_ds):,}")
        if args.resume_ckpt:
            log(f"[resume] loaded {ckpt_path} (epoch={start_epoch}, best_val_acc={best_val_acc_init})")

        history = train_model(
            model,
            train_loader,
            val_loader,
            epochs=args.epochs,
            lr=1e-4,
            device=device,
            weight_decay=1e-4,
            show_progress=True,
            checkpoint_dir=str(output_dir / "checkpoints"),
            save_every=2, 
            start_epoch=start_epoch,
            optimizer_state=optimizer_state,
            best_val_acc_init=best_val_acc_init,
            log_fn=log,
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
        summary = f"[val] loss={val_loss:.4f} acc={val_acc:.3f}"
        print(summary)
        log(summary)

    plot_training_curves(history, save_path=str(output_dir / "training_curves.png"))
    plot_precision_recall_curves(y_true, y_prob, class_names, save_path=str(output_dir / "precision_recall.png"))

    gradcam_dir = output_dir / "gradcam"
    generate_gradcam_overlays(model, val_loader, class_names, gradcam_dir, device=device, max_images=len(class_names))


if __name__ == "__main__":
    main()
