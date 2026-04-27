#!/usr/bin/env python3

"""Evaluate per-epoch checkpoints on val AND test, plot minority-class performance curves.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv
load_dotenv()

import matplotlib.pyplot as plt
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from experiments.dataset_creation.dataset_assembler import HybridDatasetAssembler, pil_to_tensor_512
from experiments.evaluation_module.classifier.resnet_classifier import build_resnet18

RUNS_DIR = PROJECT_ROOT / "runs"
RAW_DIR = PROJECT_ROOT / "results" / "epoch_progression"
FIG_DIR = PROJECT_ROOT / "experiments" / "figures"

TARGET_CLASSES = ["toaster", "hair drier"]
SPLITS = ["val", "test"]

MODEL_RUNS = [
    "baseline_model_A",
    "baseline_model_B",
    "experimental_model_A",
    "experimental_model_B",
]

DISPLAY_NAMES = {
    "baseline_model_A": "Baseline (Real Only)",
    "baseline_model_B": "Baseline (Classical Aug)",
    "experimental_model_A": "Experimental (IP-Adapter)",
    "experimental_model_B": "Experimental (ControlNet)",
}

MODEL_COLOURS = {
    "baseline_model_A": "#2196F3",      # blue
    "baseline_model_B": "#00ACC1",      # teal blue
    "experimental_model_A": "#E53935",   # red
    "experimental_model_B": "#FF9800",   # orange
}

MODEL_MARKERS = {
    "baseline_model_A": "o",
    "baseline_model_B": "s",
    "experimental_model_A": "^",
    "experimental_model_B": "D",
}

MODEL_ALPHA = {
    "baseline_model_A": 0.45,
    "baseline_model_B": 0.85,
    "experimental_model_A": 1.0,
    "experimental_model_B": 1.0,
}


def discover_epoch_checkpoints(run_dir: Path) -> List[Tuple[int, Path]]:
    """Return sorted list of (epoch_number, ckpt_path)."""
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return []
    pairs = []
    for f in ckpt_dir.iterdir():
        m = re.match(r"epoch_(\d+)\.pt", f.name)
        if m:
            pairs.append((int(m.group(1)), f))
    pairs.sort(key=lambda x: x[0])
    return pairs


def load_model_from_ckpt(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_resnet18(num_classes=80)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def get_image_paths(split: str) -> List[Path]:
    """Get sorted list of image paths for a given split."""
    split_dir = PROJECT_ROOT / "coco_dataset" / "contextual_crops" / "images" / split
    return sorted([p for p in split_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])


def evaluate_checkpoint(
    model: torch.nn.Module,
    image_paths: List[Path],
    labels: Dict[str, int],
    class_names: List[str],
    device: torch.device,
    batch_size: int = 32,
    desc: str = "Evaluating",
) -> Dict[str, Dict[str, float]]:
    """Run inference and return per-class metrics for TARGET_CLASSES only."""
    num_classes = len(class_names)
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes
    support = [0] * num_classes

    batch_tensors: List[torch.Tensor] = []
    batch_targets: List[int] = []

    def flush():
        nonlocal batch_tensors, batch_targets
        if not batch_tensors:
            return
        batch = torch.cat(batch_tensors, dim=0).to(device)
        with torch.no_grad():
            logits = model(batch)
            pred_idx = torch.argmax(logits, dim=1)
        for i in range(len(batch_targets)):
            gt = batch_targets[i]
            pi = int(pred_idx[i])
            support[gt] += 1
            if pi == gt:
                tp[gt] += 1
            else:
                fp[pi] += 1
                fn[gt] += 1
        batch_tensors = []
        batch_targets = []

    for path in tqdm(image_paths, desc=desc, unit="img", leave=False):
        gt_idx = labels.get(path.name)
        if gt_idx is None:
            continue
        img = Image.open(path).convert("RGB")
        tensor = pil_to_tensor_512(img).unsqueeze(0)
        batch_tensors.append(tensor)
        batch_targets.append(int(gt_idx))
        if len(batch_tensors) >= batch_size: flush()

    flush()

    results: Dict[str, Dict[str, float]] = {}
    for cls_name in TARGET_CLASSES:
        if cls_name not in class_names:
            continue
        c = class_names.index(cls_name)
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        rec  = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        results[cls_name] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support[c],
        }
    return results


def raw_json_path(model_name: str, split: str) -> Path:
    return RAW_DIR / f"{model_name}_{split}.json"


def load_cached_results(model_name: str, split: str) -> Dict[int, Dict[str, Dict[str, float]]] | None:
    """Load previously saved results JSON if it exists, otherwise None."""
    path = raw_json_path(model_name, split)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # Keys are stringified epoch numbers — convert back to int
    return {int(k): v for k, v in data.items()}


def save_results(model_name: str, split: str, epoch_results: Dict[int, Dict[str, Dict[str, float]]]) -> None:
    """Persist epoch results as JSON."""
    path = raw_json_path(model_name, split)
    serializable = {str(e): v for e, v in epoch_results.items()}
    path.write_text(json.dumps(serializable, indent=2))
    print(f"  [saved] {path}")


def plot_metric_for_class(
    all_data: Dict[str, Dict[int, Dict[str, Dict[str, float]]]],
    cls_name: str,
    metric: str,
    split: str,
    out_path: Path,
) -> None:
    """Single-class, single-metric line graph across all models."""
    fig, ax = plt.subplots(figsize=(8, 5))

    all_values = []
    for model_name in MODEL_RUNS:
        if model_name not in all_data:
            continue
        epoch_data = all_data[model_name]
        epochs = sorted(epoch_data.keys())
        values = []
        for e in epochs:
            cls_metrics = epoch_data[e].get(cls_name, {})
            val = cls_metrics.get(metric, 0.0)
            values.append(val)
            all_values.append(val)

        ax.plot(
            epochs, values,
            marker=MODEL_MARKERS[model_name],
            color=MODEL_COLOURS[model_name],
            label=DISPLAY_NAMES[model_name],
            alpha=MODEL_ALPHA[model_name],
            linewidth=2.5,
            markersize=10,
        )

    max_val = max(all_values) if all_values else 0.0
    upper_limit = max(max_val * 1.25, 0.1)  # 25% buffer, min 0.1
    ax.set_ylim(-0.02, min(upper_limit, 1.05))

    ax.set_xlabel("Epoch", fontsize=15)
    ax.set_ylabel(metric.capitalize(), fontsize=15)
    ax.set_title(f"{cls_name} — {metric.capitalize()} ({split})", fontsize=18, fontweight="bold")
    ax.legend(fontsize=14, markerscale=1.2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  [plot] {out_path.name}")


def plot_summary_2x2(
    full_data: Dict[str, Dict[str, Dict[int, Dict[str, Dict[str, float]]]]],
    out_path: Path,
) -> None:
    """Single 2x2 figure: Rows [Val, Test], Cols [Toaster, Hair Drier]."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
    all_f1s = []
    for split in SPLITS:
        for model_name in MODEL_RUNS:
            if split in full_data and model_name in full_data[split]:
                for epoch in full_data[split][model_name]:
                    for cls_name in TARGET_CLASSES:
                        all_f1s.append(full_data[split][model_name][epoch].get(cls_name, {}).get("f1", 0.0))
    
    max_f1 = max(all_f1s) if all_f1s else 0.0
    upper_limit = max(max_f1 * 1.3, 0.1) # 30% headroom for legend

    for row_idx, split in enumerate(SPLITS):
        for col_idx, cls_name in enumerate(TARGET_CLASSES):
            ax = axes[row_idx, col_idx]
            
            if split not in full_data:
                continue
                
            for model_name in MODEL_RUNS:
                if model_name not in full_data[split]:
                    continue
                
                epoch_data = full_data[split][model_name]
                epochs = sorted(epoch_data.keys())
                values = [epoch_data[e].get(cls_name, {}).get("f1", 0.0) for e in epochs]

                ax.plot(
                    epochs, values,
                    marker=MODEL_MARKERS[model_name],
                    color=MODEL_COLOURS[model_name],
                    label=DISPLAY_NAMES[model_name],
                    alpha=MODEL_ALPHA[model_name],
                    linewidth=3.0,
                    markersize=11,
                )

            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.02, min(upper_limit, 1.05))
            ax.tick_params(axis="both", labelsize=13)
            
            # Labels
            if row_idx == 0:
                ax.set_title(f"{cls_name} (Validation)", fontsize=20, fontweight="bold")
            else:
                ax.set_title(f"{cls_name} (Test)", fontsize=20, fontweight="bold")
                ax.set_xlabel("Epoch", fontsize=18)
            
            if col_idx == 0:
                ax.set_ylabel("F1 Score", fontsize=18)
            
            if row_idx == 0 and col_idx == 1:
                ax.legend(fontsize=14, loc="upper right", markerscale=1.2)

    fig.suptitle("Minority-Class F1 Progression: Validation vs Test", fontsize=24, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Summary f1 figure saved to {out_path}")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    env_device = os.getenv("DEVICE")
    if env_device:
        device = torch.device(env_device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"[device] using {device}")
    assembler = HybridDatasetAssembler(contextual_root=str(PROJECT_ROOT / "coco_dataset" / "contextual_crops"))
    from pycocotools.coco import COCO
    ann_path, _ = assembler._paths_for_split("train")
    coco = COCO(str(ann_path))
    assembler._build_category_mapping(coco)
    class_names = assembler.idx_to_name

    print(f"[target] tracking classes: {TARGET_CLASSES}")
    print(f"[splits] {SPLITS}")
    print()
    # full_results[split][model_name][epoch] = metrics
    full_results: Dict[str, Dict[str, Dict[int, Dict[str, Dict[str, float]]]]] = {}

    for split in SPLITS:
        print(f"  SPLIT: {split}")

        split_labels = assembler.get_image_to_label_mapping(split)
        image_paths = get_image_paths(split)
        print(f"[data] {len(image_paths):,} {split} images\n")

        split_fig_dir = FIG_DIR / split
        split_fig_dir.mkdir(parents=True, exist_ok=True)

        all_data: Dict[str, Dict[int, Dict[str, Dict[str, float]]]] = {}

        for model_name in MODEL_RUNS:
            cached = load_cached_results(model_name, split)
            if cached is not None:
                print(f"[CACHE] {model_name}/{split} — loaded from {raw_json_path(model_name, split).name}")
                all_data[model_name] = cached
                continue

            run_dir = RUNS_DIR / model_name
            checkpoints = discover_epoch_checkpoints(run_dir)
            if not checkpoints:
                print(f"[SKIP] {model_name} — no epoch checkpoints found")
                continue

            print(f"  {model_name}/{split}: {len(checkpoints)} checkpoints")

            epoch_results: Dict[int, Dict[str, Dict[str, float]]] = {}

            for epoch, ckpt_path in checkpoints:
                desc = f"  {model_name} epoch {epoch:03d} ({split})"
                model = load_model_from_ckpt(ckpt_path, device)
                metrics = evaluate_checkpoint(
                    model, image_paths, split_labels, class_names, device,
                    desc=desc,
                )
                epoch_results[epoch] = metrics

                summary_parts = []
                for cls_name in TARGET_CLASSES:
                    m = metrics.get(cls_name, {})
                    summary_parts.append(
                        f"{cls_name}: P={m.get('precision', 0):.3f} R={m.get('recall', 0):.3f} F1={m.get('f1', 0):.3f}"
                    )
                print(f"  epoch {epoch:03d} | {' | '.join(summary_parts)}")

                # free gpu memory
                del model
                if device.type == "mps":
                    torch.mps.empty_cache()
                elif device.type == "cuda":
                    torch.cuda.empty_cache()

            all_data[model_name] = epoch_results
            save_results(model_name, split, epoch_results)
            print()

        full_results[split] = all_data

        print(f"  Generating class plots ({split})")

        for cls_name in TARGET_CLASSES:
            safe_name = cls_name.replace(" ", "_")
            for metric in ("f1", "precision", "recall"):
                out_path = split_fig_dir / f"{safe_name}_{metric}.png"
                plot_metric_for_class(all_data, cls_name, metric, split, out_path)
        print()

    print("  Generating Summary 2x2 F1 Plot")
    plot_summary_2x2(full_results, FIG_DIR / "f1_val_test_progression.png")
    print(f"\n> DONE. Results saved under: {RAW_DIR}")


if __name__ == "__main__":
    main()
