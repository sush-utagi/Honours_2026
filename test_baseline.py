#!/usr/bin/env python3
"""
Evaluate the baseline classifier on the full validation set.

Given a checkpoint and COCO-style val annotations, this script:
  * loads the model
  * runs inference over every labeled validation image
  * writes a text report with overall accuracy + per-class precision/recall/F1

Outputs are saved under results/<timestamp>_baseline_report.txt by default.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from PIL import Image
from tqdm import tqdm

# Encourage MPS to fall back gracefully when ops aren't supported.
# os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

# ROOT = Path(__file__).resolve().parent
# sys.path.append(str(ROOT / "dataset_creation"))
# sys.path.append(str(ROOT / "evaluation_module" / "classifier"))

from dataset_creation.dataset_assembler import HybridDatasetAssembler, _pil_to_tensor_512  # noqa: E402
from evaluation_module.classifier.resnet_classifier import build_resnet18  # noqa: E402


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

    assembler = HybridDatasetAssembler(contextual_root="coco_dataset/contextual_crops")
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

    for p, t in zip(preds, targets):
        support[t] += 1
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

    return {
        "accuracy": accuracy,
        "per_class": per_class,
        "macro": (macro_prec, macro_rec, macro_f1),
        "support": total,
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
    lines.append("Per-class metrics:")
    lines.append(f"{'Idx':>3} {'Class':25s} {'Support':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    for idx, (prec, rec, f1, sup) in enumerate(metrics["per_class"]):
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline classifier on full validation set.")
    parser.add_argument(
        "--ckpt",
        required=True,
        help="Path to checkpoint (e.g., runs/baseline_model_A/checkpoints/best.pt)",
    )
    parser.add_argument(
        "--val-dir",
        default="coco_dataset/contextual_crops/images/val",
        help="Validation images directory.",
    )
    parser.add_argument(
        "--val-ann",
        default="coco_dataset/contextual_crops/annotations/single_instances_val.json",
        help="COCO annotations file matching --val-dir.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory to store the text report (default: results).",
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

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    val_dir = Path(args.val_dir)
    val_ann = Path(args.val_ann)
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")
    if not val_ann.exists():
        raise FileNotFoundError(f"Validation annotations not found: {val_ann}")

    class_names = load_class_names()
    val_labels = load_val_labels(val_ann)

    model = load_model(ckpt_path, device)

    preds: List[int] = []
    targets: List[int] = []

    # Track worst mistakes per class (up to 3 examples with highest wrong prob).
    worst: Dict[int, List[Tuple[str, str, float]]] = defaultdict(list)

    image_paths = [p for p in val_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}]
    if not image_paths:
        raise RuntimeError(f"No images found under {val_dir}")

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
    out_path = Path(args.results_dir) / f"{timestamp}_baseline_report.txt"
    save_report(out_path, class_names, metrics, worst)


if __name__ == "__main__":  
    main()
