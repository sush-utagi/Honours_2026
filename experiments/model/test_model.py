#!/usr/bin/env python3
"""
Evaluate the baseline or experimental classifier on the full validation set.

Given a checkpoint and COCO-style val annotations, this script:
  * loads the model
  * runs inference over every labeled validation image
  * writes a text report with overall accuracy + per-class precision/recall/F1

Outputs are saved under results/<timestamp>_<model-type>_report.txt by default.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from dataset_creation.dataset_assembler import HybridDatasetAssembler, _pil_to_tensor_512  # noqa: E402
from evaluation_module.classifier.resnet_classifier import build_resnet18  # noqa: E402

DEFAULT_VAL_DIR = PROJECT_ROOT / "coco_dataset" / "contextual_crops" / "images" / "val"
DEFAULT_VAL_ANN = PROJECT_ROOT / "coco_dataset" / "contextual_crops" / "annotations" / "single_instances_val.json"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"


def resolve_project_path(path_like: str | Path) -> Path:
    """Return absolute paths, resolving relative ones from the project root."""

    path = Path(path_like)
    return path if path.is_absolute() else PROJECT_ROOT / path


def select_device() -> torch.device:
    """Pick MPS, CUDA, then CPU."""

    use_mps = os.getenv("USE_MPS", "1") == "1"
    if use_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_resnet18(num_classes=80)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def load_class_names() -> List[str]:
    """Rebuild COCO label list without loading images."""

    from pycocotools.coco import COCO

    assembler = HybridDatasetAssembler(contextual_root=str(PROJECT_ROOT / "coco_dataset" / "contextual_crops"))
    ann_path, _ = assembler._paths_for_split("train")
    coco = COCO(str(ann_path))
    assembler._build_category_mapping(coco)
    return assembler.idx_to_name


def load_val_labels(val_ann_path: Path) -> Dict[str, int]:
    """Map image file_name -> label_idx using largest-area annotation."""

    from pycocotools.coco import COCO

    coco = COCO(str(val_ann_path))

    # category id -> contiguous idx (sorted by cat id)
    cats_sorted = sorted(coco.loadCats(coco.getCatIds()), key=lambda c: c["id"])
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(cats_sorted)}

    def primary_category(anns: List[dict]) -> int:
        def area(a: dict) -> float:
            if "area" in a:
                return float(a["area"])
            bbox = a.get("bbox", [0, 0, 0, 0])
            return float(bbox[2] * bbox[3])

        return max(anns, key=area)["category_id"]

    mapping: Dict[str, int] = {}
    for img_id, img_info in coco.imgs.items():
        file_name = img_info["file_name"]
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)
        if not anns:
            continue
        cat_id = primary_category(anns)
        label_idx = cat_id_to_idx.get(cat_id)
        if label_idx is not None:
            mapping[file_name] = label_idx
    return mapping


def predict_single(
    model: torch.nn.Module,
    device: torch.device,
    image_path: Path,
) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    tensor = _pil_to_tensor_512(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
    return logits.squeeze(0)


def compute_metrics(preds: Sequence[int], targets: Sequence[int], num_classes: int):
    """Compute accuracy and per-class precision/recall/F1 without sklearn."""

    assert len(preds) == len(targets)
    total = len(preds)
    correct = sum(int(p == t) for p, t in zip(preds, targets))
    accuracy = correct / total if total else 0.0

    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes
    support = [0] * num_classes

    full_cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

    for p, t in zip(preds, targets):
        support[t] += 1
        full_cm[t][p] += 1
        if p == t:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    per_class = []
    for c in range(num_classes):
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class.append((prec, rec, f1, support[c]))

    macro_prec = sum(pc[0] for pc in per_class) / num_classes
    macro_rec = sum(pc[1] for pc in per_class) / num_classes
    macro_f1 = sum(pc[2] for pc in per_class) / num_classes

    confusion_matrices = []
    for c in range(num_classes):
        tn = total - (tp[c] + fp[c] + fn[c])
        confusion_matrices.append((tn, fp[c], fn[c], tp[c]))

    return {
        "accuracy": accuracy,
        "per_class": per_class,
        "macro": (macro_prec, macro_rec, macro_f1),
        "support": total,
        "confusion_matrices": confusion_matrices,
        "full_cm": full_cm,
    }


def save_report(
    out_path: Path,
    class_names: Sequence[str],
    metrics: dict,
    topk_examples: Dict[int, List[Tuple[str, str, float]]],
) -> None:
    lines = []
    lines.append(f"Total samples: {metrics['support']}")
    lines.append(f"Accuracy: {metrics['accuracy']:.4f}")
    mp, mr, mf1 = metrics["macro"]
    lines.append(f"Macro Precision: {mp:.4f}")
    lines.append(f"Macro Recall:    {mr:.4f}")
    lines.append(f"Macro F1:        {mf1:.4f}")
    lines.append("")
    lines.append("Per-class metrics (sorted by F1 desc):")
    lines.append(f"{'Idx':>3} {'Class':25s} {'Support':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    sorted_per_class = sorted(
        enumerate(metrics["per_class"]), key=lambda item: item[1][2], reverse=True
    )
    for idx, (prec, rec, f1, sup) in sorted_per_class:
        cname = class_names[idx] if idx < len(class_names) else f"class_{idx}"
        lines.append(f"{idx:3d} {cname:25s} {sup:7d} {prec:7.4f} {rec:7.4f} {f1:7.4f}")

    lines.append("")
    lines.append("Top errors per class (up to 3):")
    for idx, examples in sorted(topk_examples.items()):
        if not examples:
            continue
        cname = class_names[idx] if idx < len(class_names) else f"class_{idx}"
        lines.append(f"- {idx:3d} {cname}:")
        for pred_name, path, prob in examples:
            lines.append(f"    pred={pred_name:20s} prob={prob:.3f} file={path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"[report] saved to {out_path}")


def plot_confusion_matrices(metrics: dict, class_names: Sequence[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # full confusion matrix
    full_cm = np.array(metrics["full_cm"])
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.matshow(full_cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    
    labels = [name if name in ["toaster", "hair drier"] else "" for name in class_names]
    
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title('Full Confusion Matrix', fontsize=14, pad=20)
    
    file_path = out_dir / "full_confusion_matrix.png"
    plt.savefig(file_path, bbox_inches='tight')
    plt.close(fig)

    # per-class confusion matrices ()s
    matrices = metrics["confusion_matrices"]
    for idx, (tn, fp, fn, tp) in enumerate(matrices):
        cname = class_names[idx] if idx < len(class_names) else f"class_{idx}"
        
        fig, ax = plt.subplots(figsize=(4, 4))
        cm = np.array([[tn, fp], [fn, tp]])
        
        ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.7)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='xx-large')
        
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Negative', 'Positive'])
        ax.set_yticklabels(['Negative', 'Positive'])
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.title(f'Confusion Matrix: {cname}', fontsize=14, pad=20)
        
        file_path = out_dir / f"{cname}.png"
        plt.savefig(file_path, bbox_inches='tight')
        plt.close(fig)
    print(f"[report] per-class and full confusion matrices saved to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline or experimental classifier on full evaluation set.")
    parser.add_argument(
        "--model-type",
        choices=["baseline", "experimental"],
        default="baseline",
        help="Model variant label; used for report naming (default: baseline).",
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        help="Path to checkpoint (e.g., runs/baseline_model_A/checkpoints/best.pt or runs/experimental_model_A/checkpoints/best.pt)",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="Which dataset split to evaluate on (default: val).",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Images directory (defaults to coco_dataset/contextual_crops/images/<split>).",
    )
    parser.add_argument(
        "--ann-file",
        default=None,
        help="COCO annotations file (defaults to coco_dataset/contextual_crops/annotations/single_instances_<split>.json).",
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory to store the text report (default: results).",
    )
    parser.add_argument(
        "--plot-confusion",
        action="store_true",
        help="Generate and save per-class 2x2 confusion matrices.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (reserved for future use).")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation (default: 32). Larger uses more GPU, smaller saves memory.",
    )
    args = parser.parse_args()

    device = select_device()
    print(f"[device] using {device}")

    ckpt_path = resolve_project_path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"[run] model_type={args.model_type} ckpt={ckpt_path}")

    data_dir = resolve_project_path(args.data_dir) if args.data_dir else PROJECT_ROOT / f"coco_dataset/contextual_crops/images/{args.split}"
    ann_file = resolve_project_path(args.ann_file) if args.ann_file else PROJECT_ROOT / f"coco_dataset/contextual_crops/annotations/single_instances_{args.split}.json"
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {data_dir}")
    if not ann_file.exists():
        raise FileNotFoundError(f"Annotations file not found: {ann_file}")

    class_names = load_class_names()
    val_labels = load_val_labels(ann_file)

    model = load_model(ckpt_path, device)

    preds: List[int] = []
    targets: List[int] = []

    # Track worst mistakes per class (up to 3 examples with highest wrong prob).
    worst: Dict[int, List[Tuple[str, str, float]]] = defaultdict(list)

    image_paths = [p for p in data_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}]
    if not image_paths:
        raise RuntimeError(f"No images found under {data_dir}")

    batch_tensors: List[torch.Tensor] = []
    batch_targets: List[int] = []
    batch_paths: List[Path] = []

    def flush_batch():
        nonlocal batch_tensors, batch_targets, batch_paths
        if not batch_tensors:
            return
        batch = torch.cat(batch_tensors, dim=0).to(device)
        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
            pred_prob, pred_idx = torch.max(probs, dim=1)

        for i in range(len(batch_targets)):
            gt_idx = batch_targets[i]
            pi = int(pred_idx[i])
            preds.append(pi)
            targets.append(gt_idx)
            if pi != gt_idx:
                pred_name = class_names[pi] if pi < len(class_names) else str(pi)
                entry = (pred_name, str(batch_paths[i]), float(pred_prob[i]))
                bucket = worst[gt_idx]
                bucket.append(entry)
                bucket.sort(key=lambda x: x[2], reverse=True)
                if len(bucket) > 3:
                    bucket.pop()

        batch_tensors = []
        batch_targets = []
        batch_paths = []

    progress = tqdm(image_paths, desc="Evaluating", unit="img")
    for path in progress:
        fname = path.name
        gt_idx = val_labels.get(fname)
        if gt_idx is None:
            continue

        img = Image.open(path).convert("RGB")
        tensor = _pil_to_tensor_512(img).unsqueeze(0)  # keep on CPU until batch
        batch_tensors.append(tensor)
        batch_targets.append(int(gt_idx))
        batch_paths.append(path)

        if len(batch_tensors) >= args.batch_size:
            flush_batch()

    flush_batch()

    metrics = compute_metrics(preds, targets, num_classes=len(class_names))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = resolve_project_path(args.results_dir) / f"{timestamp}_{args.model_type}_report.txt"
    save_report(out_path, class_names, metrics, worst)

    if args.plot_confusion:
        cm_out_dir = resolve_project_path(args.results_dir) / f"{timestamp}_{args.model_type}_{args.split}_cm"
        plot_confusion_matrices(metrics, class_names, cm_out_dir)


if __name__ == "__main__":  
    main()
