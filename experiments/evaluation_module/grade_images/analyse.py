#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image
from torchvision import transforms

import matplotlib
matplotlib.use("Agg")  # headless backend no display needed
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.stats import wasserstein_distance

from experiments.evaluation_module.grade_images.grader import (
    compute_scores,
    compute_scores_per_prompt,
    extract_clip_embeddings,
    extract_clip_text_embeddings,
)


def _repo_root() -> Path: return Path(__file__).resolve().parents[3]


def _collect_real_paths(
    class_name: str,
    contextual_root: Path,
    split: str = "train",
    max_images: Optional[int] = None,
) -> List[Path]:
    ROOT = Path(__file__).resolve().parents[4]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
        
    from experiments.dataset_creation.dataset_assembler import HybridDatasetAssembler
    
    print(f"[analyse] finding {split} images for class '{class_name}' using HybridDatasetAssembler …")
    assembler = HybridDatasetAssembler(contextual_root=str(contextual_root))
    paths_dict = assembler.get_image_paths_by_class(
        split=split,
        target_classes=[class_name],
        max_samples=max_images
    )
    
    paths = paths_dict.get(class_name, [])
    print(f"[analyse] found {len(paths):,} {split} images for class '{class_name}'")
    return paths


def _load_prompt_json(
    prompt_json_path: Path,
) -> Tuple[Dict[str, str], List[dict]]: # (filename_to_prompt, samples)
    print(f"[analyse] loading prompts from {prompt_json_path}")
    with prompt_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data.get("samples", [])
    if not samples:
        print("[analyse] warning: prompt JSON contains no samples.")
        return {}, []

    filename_to_prompt: Dict[str, str] = {}
    for sample in samples:
        init_img = sample.get("init_image", "")
        if init_img:
            filename_to_prompt[Path(init_img).name] = sample.get("prompt", "")

    return filename_to_prompt, samples


def _pair_synthetic_with_samples(
    synthetic_paths: List[Path],
    samples: List[dict],
) -> Optional[List[dict]]:
    """Map each synthetic image to its full JSON-sample metadata (for init_image lookup)."""
    if len(synthetic_paths) != len(samples):
        print(
            f"[analyse] warning: {len(synthetic_paths)} images vs "
            f"{len(samples)} JSON samples — skipping conditioning fidelity analysis."
        )
        return None
    return samples


def _compute_metrics(scores: List[float]) -> Dict[str, float]:
    arr = np.array(scores, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return {"mean": float("nan"), "median": float("nan"), "std": float("nan"), "count": 0}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "count": int(len(arr)),
    }


def _compute_pairwise_diversity(embeddings: np.ndarray) -> Dict[str, float]:
    """Calculate mean pairwise cosine distance (expects L2-normalized inputs)."""
    n = len(embeddings)
    if n < 2: return {"mean": float("nan")}
    
    # Similarity matrix (N x N) - Dot product is cosine similarity if normalized
    sim_matrix = embeddings @ embeddings.T
    
    tri_i, tri_j = np.triu_indices(n, k=1)
    dists = 1.0 - sim_matrix[tri_i, tri_j]
    return {"mean": float(np.mean(dists)), "count": int(len(dists))}


def _compute_conditioning_fidelity(
    synth_paths: List[Path],
    source_paths: List[Path],
    image_size: Tuple[int, int] = (64, 64),
) -> Dict[str, object]:
    """Measure the 1-to-1 pixel-space distance (RMSE) between outputs and their sources."""
    num_elements = image_size[0] * image_size[1] * 3
    
    def load_flat(path: Path) -> Optional[np.ndarray]:
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize(image_size)
            return np.array(img).flatten().astype(np.float32) / 255.0
        except Exception: return None

    dists = []
    for s_p, r_p in zip(synth_paths, source_paths):
        s_v, r_v = load_flat(s_p), load_flat(r_p)
        if s_v is not None and r_v is not None:
            # We calculate RMSE (Root Mean Square Error) 
            # This normalizes the L2 norm by the count of pixels so it scales [0, 1]
            rmse = np.linalg.norm(s_v - r_v) / np.sqrt(num_elements)
            dists.append(float(rmse))
            
    return {
        "distances": dists,
        "mean_dist": float(np.mean(dists)) if dists else 0,
        "median_dist": float(np.median(dists)) if dists else 0,
    }


def _compute_clip_fidelity(
    synth_embeds: np.ndarray,
    source_embeds: np.ndarray,
) -> Dict[str, object]:
    """Measure the 1-to-1 semantic distance between outputs and their source images."""
    if len(synth_embeds) != len(source_embeds):
        return {"mean_dist": float("nan"), "distances": []}
    
    # Cosine distance = 1 - dot product (assuming normalized)
    dists = 1.0 - np.sum(synth_embeds * source_embeds, axis=1)
    
    return {
        "distances": [float(d) for d in dists],
        "mean_dist": float(np.mean(dists)),
        "median_dist": float(np.median(dists)),
    }


def _compute_memorization_ratio_pixel(
    synth_paths: List[Path],
    source_paths: List[Path],
    train_paths: List[Path],
    image_size: Tuple[int, int] = (64, 64),
) -> Dict[str, object]:
    """Compute R(x_syn) = d(x_syn, x_src) / d(x_syn, x_src⊥) in pixel space.

    x_src⊥ is the nearest training-set image that is NOT the source.
    R ≪ 1 -> memorization, R ≈ 1 -> meaningful transformation, R > 1 -> potential mode drift.
    """

    def _load_flat(path: Path) -> Optional[np.ndarray]:
        try:
            img = Image.open(path).convert("RGB").resize(image_size)
            return np.array(img).flatten().astype(np.float32) / 255.0
        except Exception:
            return None

    num_elements = image_size[0] * image_size[1] * 3

    # Pre-load a subsample of training images as x_src⊥ pool
    max_pool = min(len(train_paths), 500)
    pool_indices = random.sample(range(len(train_paths)), max_pool) if len(train_paths) > max_pool else list(range(len(train_paths)))
    pool_vecs: List[Tuple[Path, np.ndarray]] = []
    for idx in pool_indices:
        v = _load_flat(train_paths[idx])
        if v is not None:
            pool_vecs.append((train_paths[idx], v))

    if not pool_vecs:
        return {"ratios": [], "mean_ratio": float("nan"), "median_ratio": float("nan")}

    pool_matrix = np.stack([v for _, v in pool_vecs])  # (P, D)
    pool_paths_list = [p for p, _ in pool_vecs]

    ratios: List[float] = []
    for s_p, src_p in zip(synth_paths, source_paths):
        s_v = _load_flat(s_p)
        src_v = _load_flat(src_p)
        if s_v is None or src_v is None:
            continue

        d_src = np.linalg.norm(s_v - src_v) / np.sqrt(num_elements)

        # Distance to every pool image
        diffs = pool_matrix - s_v[np.newaxis, :]  # (P, D)
        pool_dists = np.linalg.norm(diffs, axis=1) / np.sqrt(num_elements)  # (P,)

        # Exclude the source itself (by path match)
        src_resolved = src_p.resolve()
        mask = np.array([pp.resolve() != src_resolved for pp in pool_paths_list])
        if not mask.any():
            continue
        d_distractor = float(np.min(pool_dists[mask]))

        if d_distractor < 1e-12:
            continue  # degenerate
        ratios.append(d_src / d_distractor)

    return {
        "ratios": ratios,
        "mean_ratio": float(np.mean(ratios)) if ratios else float("nan"),
        "median_ratio": float(np.median(ratios)) if ratios else float("nan"),
    }


def _compute_memorization_ratio_clip(
    synth_embeds: np.ndarray,
    source_embeds: np.ndarray,
    train_embeds: np.ndarray,
    source_indices_in_train: Optional[List[int]] = None,
) -> Dict[str, object]:
    """Compute R(x_syn) = d_CLIP(x_syn, x_src) / d_CLIP(x_syn, x_src⊥) in CLIP space.

    x_src⊥ is the nearest training-set image that is NOT the source.
    All embeddings must be L2-normalised so dot product = cosine similarity.
    R ≪ 1 -> memorization, R ≈ 1 -> meaningful transformation, R > 1 -> mode drift.
    """
    n = len(synth_embeds)
    if n == 0 or len(source_embeds) != n:
        return {"ratios": [], "mean_ratio": float("nan"), "median_ratio": float("nan")}

    # d(syn, src) — paired cosine distance
    d_src = 1.0 - np.sum(synth_embeds * source_embeds, axis=1)  # (N,)

    # d(syn, all_train) — full similarity matrix
    sim_matrix = synth_embeds @ train_embeds.T  # (N, T)
    dist_matrix = 1.0 - sim_matrix  # cosine distance

    # For each synthetic image, find the nearest distractor (exclude source)
    ratios: List[float] = []
    for i in range(n):
        row = dist_matrix[i].copy()  # (T,)

        # Mask out the source if we know its index in train_embeds
        if source_indices_in_train is not None and source_indices_in_train[i] is not None:
            row[source_indices_in_train[i]] = np.inf
        else:
            # Heuristic: mask any training image whose embedding is identical
            # (within tolerance) to the source embedding
            source_sim = train_embeds @ source_embeds[i]
            identical_mask = source_sim > 0.9999
            row[identical_mask] = np.inf

        d_distractor = float(np.min(row))
        if d_distractor < 1e-12:
            continue  # degenerate
        ratios.append(float(d_src[i]) / d_distractor)

    return {
        "ratios": ratios,
        "mean_ratio": float(np.mean(ratios)) if ratios else float("nan"),
        "median_ratio": float(np.median(ratios)) if ratios else float("nan"),
    }


def _compute_lpips_diversity(
    image_paths: List[Path],
    n_bootstrap: int = 150,
    device: str = "cpu",
) -> Dict[str, float]:
    """Mean LPIPS distance over random pairs (bootstrapped)."""
    if len(image_paths) < 2: return {"mean": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan")}
    import lpips
    import warnings

    # Mute LPIPS setup output
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Ensure stdout isn't polluted by LPIPS initialization
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            loss_fn = lpips.LPIPS(net='alex', verbose=False).to(device)
        finally:
            sys.stdout = old_stdout

    # LPIPS normalizes input strictly to [-1, 1]
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    distances = []
    with torch.no_grad():
        for _ in range(n_bootstrap):
            p1, p2 = random.sample(image_paths, 2)
            try:
                img1 = Image.open(p1).convert("RGB")
                img2 = Image.open(p2).convert("RGB")
                t1 = transform(img1).unsqueeze(0).to(device)
                t2 = transform(img2).unsqueeze(0).to(device)
                
                dist = loss_fn(t1, t2).item()
                distances.append(dist)
            except Exception:
                continue

    if not distances:
        return {"mean": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan")}

    return {
        "mean": float(np.mean(distances)),
        "ci_lower": float(np.percentile(distances, 2.5)),
        "ci_upper": float(np.percentile(distances, 97.5)),
        "n_bootstrap": len(distances),
    }


def _save_histogram(
    real_scores: Optional[List[float]],
    synthetic_scores: Optional[List[float]],
    class_name: str,
    out_path: Path,
    technique: Optional[str] = None,
) -> None:
    """Plot the overlap of Real and Synthetic CLIP score distributions using dual Y-axes."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, 45, 51)
    
    if real_scores:
        real_clean = [s for s in real_scores if not math.isnan(s)]
        ax1.hist(real_clean, bins=bins, alpha=0.5, color="#4A90D9", label="Real", edgecolor="#2A5079", linewidth=0.5)
        ax1.set_ylabel("Count (Real)", color="#4A90D9", fontsize=12, fontweight="bold")
        ax1.tick_params(axis='y', labelcolor="#4A90D9")

    if synthetic_scores:
        ax2 = ax1.twinx()
        synth_clean = [s for s in synthetic_scores if not math.isnan(s)]
        ax2.hist(synth_clean, bins=bins, alpha=0.4, color="#E8833A", label="Synthetic", edgecolor="#A65D29", linewidth=0.5)
        ax2.set_ylabel("Count (Synthetic)", color="#E8833A", fontsize=12, fontweight="bold")
        ax2.tick_params(axis='y', labelcolor="#E8833A")
        
        # Combine legends from both axes
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper left", fontsize=11, framealpha=0.9)
    else:
        ax1.legend(loc="upper left", fontsize=11)

    tech_label = f" ({technique})" if technique else ""
    ax1.set_xlabel("CLIP Score (0–100)", fontsize=12)
    ax1.set_title(f"CLIP Score Distribution — {class_name}{tech_label}", fontsize=14, fontweight="bold", pad=15)
    ax1.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[analyse] saved dual-axis histogram -> {out_path}")


def _save_metrics_json(
    real_domain: Optional[Dict[str, float]],
    synth_domain: Optional[Dict[str, float]],
    synth_prompt: Optional[Dict[str, float]],
    emd: Optional[float],
    class_name: str,
    out_path: Path,
) -> None:
    """Write metrics_summary.json."""
    payload: Dict = {"class": class_name}

    if real_domain is not None:
        payload["real"] = {"domain_level": real_domain}
    if synth_domain is not None:
        payload["synthetic"] = {"domain_level": synth_domain}
        if synth_prompt is not None:
            payload["synthetic"]["prompt_level"] = synth_prompt
    if emd is not None:
        payload["wasserstein_distance_domain"] = emd

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[analyse] saved metrics  -> {out_path}")


def _save_failure_cases(
    synthetic_paths: List[Path],
    domain_scores: List[float],
    prompt_scores: Optional[List[float]],
    prompts: List[str],
    out_path: Path,
    n: int = 10,
) -> None:
    """Write ``failure_cases.csv`` — bottom *n* synthetic images by domain CLIP score."""
    indexed = [
        (i, domain_scores[i], synthetic_paths[i])
        for i in range(len(synthetic_paths))
        if not math.isnan(domain_scores[i])
    ]
    indexed.sort(key=lambda t: t[1])
    worst = indexed[:n]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "filename", "domain_clip_score", "prompt_clip_score", "prompt"])
        for rank, (i, dscore, path) in enumerate(worst, start=1):
            pscore = prompt_scores[i] if prompt_scores else ""
            prompt_text = prompts[i] if i < len(prompts) else ""
            writer.writerow([rank, path.name, f"{dscore:.6f}", f"{pscore:.6f}" if pscore != "" else "", prompt_text])

    print(f"[analyse] saved failures -> {out_path}")


def _save_extremes(
    image_paths: List[Path],
    scores: List[float],
    source_label: str,
    class_name: str,
    out_dir: Path,
    n: int = 5,
) -> None:
    indexed = [
        (i, scores[i]) for i in range(len(image_paths))
        if not math.isnan(scores[i])
    ]
    indexed.sort(key=lambda t: t[1])

    if len(indexed) < n:
        print(f"[analyse] not enough valid images for {source_label} extremes (need {n}, have {len(indexed)})")
        return

    worst = indexed[:n]
    best = indexed[-n:][::-1]  # highest first

    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 9.0))
    fig.suptitle(f"{class_name} — {source_label} extremes (domain CLIP score)", fontsize=14, fontweight="bold", y=0.98)

    for col, (idx, score) in enumerate(best):
        ax = axes[0, col]
        img = plt.imread(str(image_paths[idx]))
        ax.imshow(img)
        ax.set_title(f"#{col+1} Best", fontsize=9, color="#2E7D32", fontweight="bold")
        ax.text(
            0.5, -0.02, f"{score:.1f}",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#C8E6C9", edgecolor="#2E7D32", alpha=0.9),
        )
        ax.axis("off")

    for col, (idx, score) in enumerate(worst):
        ax = axes[1, col]
        img = plt.imread(str(image_paths[idx]))
        ax.imshow(img)
        ax.set_title(f"#{col+1} Worst", fontsize=9, color="#C62828", fontweight="bold")
        ax.text(
            0.5, -0.02, f"{score:.1f}",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFCDD2", edgecolor="#C62828", alpha=0.9),
        )
        ax.axis("off")

    fig.tight_layout(rect=[0, 0.02, 1, 0.95], h_pad=3.0)
    out_path = out_dir / f"{source_label}_extremes.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[analyse] saved extremes -> {out_path}")


def _run_umap(
    embeddings: np.ndarray, # (N, D) array of L2-normalised CLIP embeddings
    n_neighbors: int = 15,
    min_dist: float = 0.2,
    seed: int = 42,
) -> np.ndarray: # (N, 2) UMAP embedding
    import umap as umap_lib
    print(
        f"[analyse] running UMAP ({embeddings.shape[0]:,} points, "
        f"{embeddings.shape[1]}D -> 2D, n_neighbors={n_neighbors}, "
        f"min_dist={min_dist}) …"
    )
    reducer = umap_lib.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric="cosine",
        random_state=seed,
    )
    result = reducer.fit_transform(embeddings)
    print(f"[analyse] UMAP complete -> {result.shape}")
    return result


def _compute_umap_metrics(
    synth_2d: np.ndarray,
    train_2d: np.ndarray,
) -> Dict[str, float]:

    # Centroids
    synth_centroid = synth_2d.mean(axis=0)
    train_centroid = train_2d.mean(axis=0)
    centroid_dist = float(np.linalg.norm(synth_centroid - train_centroid))

    # Spread: mean distance from centroid
    synth_spread = float(np.mean(np.linalg.norm(synth_2d - synth_centroid, axis=1)))
    train_spread = float(np.mean(np.linalg.norm(train_2d - train_centroid, axis=1)))
    spread_ratio = synth_spread / max(train_spread, 1e-8)

    # Overlap: fraction of synthetic points inside the train convex hull
    overlap = 0.0
    try:
        if len(train_2d) >= 4:
            hull = Delaunay(train_2d)
            overlap = float(np.mean(hull.find_simplex(synth_2d) >= 0))
    except Exception:
        pass

    return {
        "centroid_distance": round(centroid_dist, 4),
        "synth_spread": round(synth_spread, 4),
        "train_spread": round(train_spread, 4),
        "spread_ratio": round(spread_ratio, 4),
        "overlap_synth_in_train": round(overlap, 4),
    }


def _save_umap_plot(
    train_2d: np.ndarray,
    val_2d: np.ndarray,
    synth_2d: np.ndarray,
    dataset_label: str,
    metrics: Dict[str, float],
    class_name: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    C_TRAIN = "#4A90D9"   # Blue
    C_VAL   = "#50C878"   # Emerald green
    C_SYNTH = "#E8833A"   # Orange
    for pts, color in [
        (train_2d, C_TRAIN),
        (val_2d,   C_VAL),
        (synth_2d, C_SYNTH),
    ]:
        if len(pts) >= 3: # otherwise cant draw convex hull 
            try:
                hull = ConvexHull(pts)
                hull_verts = np.append(hull.vertices, hull.vertices[0])
                ax.fill(pts[hull_verts, 0], pts[hull_verts, 1], alpha=0.08, color=color, zorder=1)
                ax.plot(pts[hull_verts, 0], pts[hull_verts, 1], color=color, alpha=0.35, linewidth=1.2, zorder=2)
            except Exception:
                pass  # degenerate hull (collinear points, etc.)
    ax.scatter(train_2d[:, 0], train_2d[:, 1], c=C_TRAIN, marker="o", s=35, alpha=0.72, edgecolors="white", linewidths=0.3, label=f"Real Train (n={len(train_2d):,})", zorder=3)
    if len(val_2d) > 0:
        ax.scatter(val_2d[:, 0], val_2d[:, 1], c=C_VAL, marker="D", s=32, alpha=0.72, edgecolors="white", linewidths=0.3, label=f"Real Val (n={len(val_2d):,})", zorder=3)
    ax.scatter(synth_2d[:, 0], synth_2d[:, 1], c=C_SYNTH, marker="^", s=40, alpha=0.72, edgecolors="white", linewidths=0.3, label=f"Synthetic (n={len(synth_2d):,})", zorder=3)
    ax.set_title(f"CLIP Embedding UMAP — {class_name}\nDataset: {dataset_label}", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("UMAP-1", fontsize=11)
    ax.set_ylabel("UMAP-2", fontsize=11)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9, edgecolor="#ddd")
    overlap_pct = metrics.get("overlap_synth_in_train", 0)
    spread_r    = metrics.get("spread_ratio", 0)
    centroid_d  = metrics.get("centroid_distance", 0)
    annot = f"Overlap (synth-train): {overlap_pct:.0%}\nSpread ratio (synth/train): {spread_r:.2f}\nCentroid distance: {centroid_d:.2f}"
    ax.text(0.98, 0.03, annot, transform=ax.transAxes, fontsize=8, verticalalignment="bottom", horizontalalignment="right", fontfamily="monospace", bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="#cccccc",))
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_fidelity_plot(
    pixel_dists: List[float],
    clip_dists: List[float],
    class_name: str,
    out_path: Path,
) -> None:
    """Save histograms showing how closely outputs followed their conditioning source.

    Uses empirical p10/p90 percentile bands rather than a hardcoded threshold.
    The Goldilocks zone (legitimate transformation) falls between p10 and p90.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Conditioning Fidelity (Source vs Output) — {class_name}", fontsize=14, fontweight="bold")

    for i, (ax, dists, label, x_lab) in enumerate(zip(
        axes,
        [pixel_dists, clip_dists],
        ["Pixel Space (RMSE)", "CLIP Space (Cosine Distance)"],
        ["RMSE", "Cosine Distance"],
    )):
        if not dists:
            ax.text(0.5, 0.5, "No pairing data", transform=ax.transAxes, ha='center')
            continue

        arr = np.array(dists)
        p10 = np.percentile(arr, 10)
        p90 = np.percentile(arr, 90)
        is_pixel = (i == 0)

        ax.hist(dists, bins=40, color="#50C878" if is_pixel else "#4DB6AC", alpha=0.75, edgecolor="white")

        # Empirical low-end: most similar to source (Bottom 10%)
        ax.axvline(p10, color="#C62828", linestyle="--",
                   label=f"p10 (Bottom 10% change): {p10:.3f}")
        
        if is_pixel:
            # Empirical high-end: most varied from source (Top 10%)
            ax.axvline(p90, color="#29B6F6", linestyle="--",
                       label=f"p90 (Highest 10% change): {p90:.3f}")
            ax.axvspan(p10, p90, alpha=0.08, color="#4CAF50", label="Central 80% zone")
            ax.axvspan(p90, max(arr)*1.05 if len(arr)>0 else p90*1.1, alpha=0.08, color="#B3E5FC", label="High variation outliers")
        else:
            # Empirical high-end: most semantic distance from source (Top 10%)
            ax.axvline(p90, color="#1565C0", linestyle="--",
                       label=f"p90 (Highest 10% semantic change): {p90:.3f}")
            ax.axvspan(p10, p90, alpha=0.08, color="#4CAF50", label="Central 80% zone")

        ax.set_xlabel(x_lab, fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"{label} Distribution", fontsize=12)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[analyse] saved fidelity plot -> {out_path}")


def _save_memorization_ratio_plot(
    pixel_ratios: List[float],
    clip_ratios: List[float],
    class_name: str,
    out_path: Path,
) -> None:
    """Save histograms of the Yoon-adapted memorization ratio R.

    R = d(x_syn, x_src) / d(x_syn, x_src⊥)
    R ≪ 1 -> memorization, R ≈ 1 -> meaningful transformation, R > 1 -> mode drift.

    Plots descriptive statistics only (median, R=1 reference, counts).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Memorization Ratio R (Yoon-adapted) — {class_name}",
        fontsize=14, fontweight="bold",
    )

    for ax, ratios, space_label in zip(
        axes,
        [pixel_ratios, clip_ratios],
        ["Pixel Space", "CLIP Space"],
    ):
        if not ratios:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue

        arr = np.array(ratios)
        ax.hist(arr, bins=40, color="#7E57C2", alpha=0.75, edgecolor="white")

        # Key reference line: R = 1 (equidistant)
        ax.axvline(1.0, color="#424242", linestyle="-", linewidth=1.2,
                   label="R = 1 (equidistant)")

        # Median of the current run
        median_r = np.median(arr)
        ax.axvline(median_r, color="#2E7D32", linestyle=":",
                   label=f"median R: {median_r:.3f}")

        # Descriptive counts
        n_below_1 = int(np.sum(arr < 1.0))
        n_above_1 = int(np.sum(arr > 1.0))
        ax.text(
            0.97, 0.95,
            f"n(R < 1) = {n_below_1}\nn(R > 1) = {n_above_1}\n"
            f"mean R = {np.mean(arr):.3f}",
            transform=ax.transAxes, fontsize=8,
            verticalalignment="top", horizontalalignment="right",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      alpha=0.85, edgecolor="#cccccc"),
        )

        ax.set_xlabel("R = d(syn, src) / d(syn, distractor)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"{space_label}", fontsize=12)
        ax.legend(fontsize=7.5, loc="upper right")
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[analyse] saved memorization ratio plot -> {out_path}")


def _save_umap_metrics(
    metrics: Dict[str, float],
    class_name: str,
    out_path: Path,
) -> None:
    payload = {
        "class": class_name,
        "metrics": metrics,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[analyse] saved UMAP metrics -> {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="CLIP score analysis for real vs synthetic images.",formatter_class=argparse.RawDescriptionHelpFormatter,)
    parser.add_argument("--class-name", default=None, help="Inferred from --synthetic-dir if omitted.")
    parser.add_argument("--source", choices=["real", "synthetic", "both"], default="both")
    parser.add_argument("--real-dir", default=None)
    parser.add_argument("--synthetic-dir", default=None, help="Path to specific run folder (e.g. data_generation_outputs/toaster_100_ip)")
    parser.add_argument("--prompt-json", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-real", type=int, default=None)
    parser.add_argument("--no-prompt-level", action="store_true")
    parser.add_argument("--no-umap", action="store_true")
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.2)

    parser.add_argument("--out-dir", default=None, help="Explicit output directory (overrides default results/clip_analysis/<class>_<suffix>).")
    args = parser.parse_args()

    repo_root = _repo_root()
    synth_dir: Optional[Path] = None
    inferred_class = args.class_name
    
    if args.synthetic_dir:
        synth_dir = Path(args.synthetic_dir)
        if not inferred_class:
            folder_parts = synth_dir.name.split('_')
            if folder_parts[0] == "hair": inferred_class = "hair drier"
            else: inferred_class = folder_parts[0]
            print(f"[analyse] inferred class '{inferred_class}' from folder '{synth_dir.name}'")

    if not inferred_class:
        print("[analyse] error: must provide --class_name or --synthetic_dir pointing to a run folder.")
        return 1

    class_name = inferred_class
    real_root = Path(args.real_dir) if args.real_dir else repo_root / "coco_dataset" / "contextual_crops"
    do_real = args.source in ("real", "both")
    do_synth = args.source in ("synthetic", "both")
    real_paths: List[Path] = []
    synth_paths: List[Path] = []

    if do_real: real_paths = _collect_real_paths(class_name, real_root, max_images=args.max_real)
    if do_synth:
        if not synth_dir or not synth_dir.exists():
            print(f"[analyse] error: synthetic directory not found: {args.synthetic_dir}")
            return 1
        synth_paths = sorted(p for p in synth_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".png"})
        print(f"[analyse] found {len(synth_paths):,} synthetic images in {synth_dir}")

    # synthetic prompt-level scores use json
    prompt_json_path = Path(args.prompt_json) if args.prompt_json else None    
    if do_synth and prompt_json_path is None and synth_dir:
        json_files = list(synth_dir.glob("*.json"))
        if json_files:
            prompt_json_path = json_files[0]
            print(f"[analyse] found prompt JSON in output folder: {prompt_json_path.name}")

    synth_prompts: List[str] = []

    if do_synth and synth_paths and not args.no_prompt_level:
        if prompt_json_path is None:
            print("[analyse] error: --prompt-json is required for synthetic image analysis (could not auto-detect one).")
            return 1
        
        filename_to_prompt, json_samples = _load_prompt_json(prompt_json_path)
        # Re-map prompts to match the order of synth_paths
        for p in synth_paths:
            synth_prompts.append(filename_to_prompt.get(p.name, ""))

    suffix = "syn"
    tech_sources = []
    if synth_dir: tech_sources.append(synth_dir.name.lower())
    if prompt_json_path: tech_sources.append(prompt_json_path.name.lower())

    for s in tech_sources:
        parts = s.split('_')
        if "ip" in parts: 
            suffix = "ip"
            break
        if "cn" in parts:
            suffix = "cn"
            break
        if "ti" in parts:
            suffix = "ti"
            break

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        class_folder = class_name.replace(" ", "_")
        base_class_dir = repo_root / "results" / "clip_analysis" / class_folder
        
        if suffix == "syn":
            out_dir = base_class_dir
        else:
            out_dir = base_class_dir / f"{class_folder}_{suffix}"

    print(f"\n[analyse] extracting CLIP embeddings for all relevant sets …")
    
    train_embeds: np.ndarray = np.empty((0, 768))
    train_valid_idx: List[int] = []
    if do_real and real_paths:
        train_embeds, train_valid_idx = extract_clip_embeddings(real_paths, batch_size=args.batch_size)
        if len(train_embeds) > 0:
            train_embeds = train_embeds / np.maximum(np.linalg.norm(train_embeds, axis=1, keepdims=True), 1e-8)



    synth_embeds: np.ndarray = np.empty((0, 768))
    synth_valid_idx: List[int] = []
    if do_synth and synth_paths:
        synth_embeds, synth_valid_idx = extract_clip_embeddings(synth_paths, batch_size=args.batch_size)
        if len(synth_embeds) > 0:
            synth_embeds = synth_embeds / np.maximum(np.linalg.norm(synth_embeds, axis=1, keepdims=True), 1e-8)
        
    val_embeds: np.ndarray = np.empty((0, 768))
    val_valid_idx: List[int] = []
    if (do_real and do_synth) and not args.no_umap:
        try:
            val_paths = _collect_real_paths(class_name, real_root, split="val", max_images=args.max_val)
            if val_paths:
                val_embeds, val_valid_idx = extract_clip_embeddings(val_paths, batch_size=args.batch_size)
        except Exception as e:
            print(f"[analyse] validation images skipped: {e}")

    # CLIP scores via dot products
    domain_text = f"a photo of a {class_name}"
    print(f"\n[analyse] scoring images against domain text: \"{domain_text}\"")

    real_domain_scores: List[float] = [float("nan")] * len(real_paths)
    synth_domain_scores: List[float] = [float("nan")] * len(synth_paths)
    synth_prompt_scores: List[float] = [float("nan")] * len(synth_paths)

    domain_text_embed = extract_clip_text_embeddings([domain_text])[0]

    if len(train_embeds) > 0:
        # Cosine similarity * 100
        sims = (train_embeds @ domain_text_embed.T) * 100
        for i, sim in zip(train_valid_idx, sims):
            real_domain_scores[i] = float(max(sim, 0))

    if len(synth_embeds) > 0:
        sims = (synth_embeds @ domain_text_embed.T) * 100
        for i, sim in zip(synth_valid_idx, sims):
            synth_domain_scores[i] = float(max(sim, 0))

        # NOTE: 
        # if not args.no_prompt_level and synth_prompts:
        #     print(f"[analyse] scoring synthetic adherence against individual prompts …")
        #     # This is still a bit heavy as it encodes N text strings, but much faster than N images
        #     prompt_embeds = extract_clip_text_embeddings(synth_prompts)
        #     
        #     # We only have embeddings for synth_valid_idx
        #     for i, actual_idx in enumerate(synth_valid_idx):
        #         s_feat = synth_embeds[i]
        #         t_feat = prompt_embeds[actual_idx]
        #         sim = (s_feat @ t_feat.T) * 100
        #         synth_prompt_scores[actual_idx] = float(max(sim, 0))

    # -- Compute metrics ------------------------------------------------------
    real_domain_metrics = _compute_metrics(real_domain_scores) if real_domain_scores else None
    synth_domain_metrics = _compute_metrics(synth_domain_scores) if synth_domain_scores else None
    synth_prompt_metrics = _compute_metrics(synth_prompt_scores) if synth_prompt_scores else None

    emd: Optional[float] = None
    if real_domain_scores and synth_domain_scores:
        real_clean = [s for s in real_domain_scores if not math.isnan(s)]
        synth_clean = [s for s in synth_domain_scores if not math.isnan(s)]
        if real_clean and synth_clean:
            emd = float(wasserstein_distance(real_clean, synth_clean))

    # -- Print summary to stdout ----------------------------------------------
    print("\n" + "=" * 60)
    print(f"  CLIP Analysis Results — {class_name}")
    print("=" * 60)
    if real_domain_metrics:
        m = real_domain_metrics
        print(f"  Real (domain)    | n={m['count']:,}  mean={m['mean']:.4f}  median={m['median']:.4f}  std={m['std']:.4f}")
    if synth_domain_metrics:
        m = synth_domain_metrics
        print(f"  Synth (domain)   | n={m['count']:,}  mean={m['mean']:.4f}  median={m['median']:.4f}  std={m['std']:.4f}")
    # if synth_prompt_metrics:
    #     m = synth_prompt_metrics
    #     print(f"  Synth (prompt)   | n={m['count']:,}  mean={m['mean']:.4f}  median={m['median']:.4f}  std={m['std']:.4f}")
    if emd is not None:
        print(f"  Wasserstein dist | {emd:.6f}")
    print("=" * 60)

    # -- Save outputs ---------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)



    # Readable technique name for plotting
    tech_map = {"ip": "IP-Adapter", "cn": "ControlNet", "ti": "Textual Inversion"}
    readable_tech = tech_map.get(suffix)

    _save_histogram(
        real_domain_scores or None,
        synth_domain_scores or None,
        class_name,
        out_dir / "distribution_histogram.png",
        technique=readable_tech
    )

    _save_metrics_json(
        real_domain_metrics,
        synth_domain_metrics,
        synth_prompt_metrics,
        emd,
        class_name,
        out_dir / "metrics_summary.json",
    )

    if do_synth and synth_domain_scores:
        _save_failure_cases(
            synth_paths,
            synth_domain_scores,
            synth_prompt_scores or None,
            synth_prompts,
            out_dir / "failure_cases.csv",
        )

    # -- Save best/worst image panels -----------------------------------------
    if do_real and real_domain_scores:
        _save_extremes(real_paths, real_domain_scores, "real", class_name, out_dir)
    if do_synth and synth_domain_scores:
        _save_extremes(synth_paths, synth_domain_scores, "synthetic", class_name, out_dir)

    # -- UMAP distributional visualization ------------------------------------
    if len(train_embeds) > 0 and len(synth_embeds) > 0 and not args.no_umap:
        print("\n" + "-" * 60)
        print("  Distributional Analysis & UMAP")
        print("-" * 60)

        # --- Pairwise Diversity Analysis ---
        print("\n[analyse] computing pairwise diversity (mode collapse check) …")
        min_len = min(len(train_embeds), len(synth_embeds))
        if min_len >= 2:
            # bounds to ensure reliable subset but avoiding massive matrix calcs
            sub_n = max(2, min(150, min_len))
            
            print("  => Semantic (CLIP) diversity:")
            real_div = _compute_pairwise_diversity(train_embeds)
            synth_div = _compute_pairwise_diversity(synth_embeds)
            print(f"     Real   : {real_div['mean']:.4f} (all pairs)")
            print(f"     Synth  : {synth_div['mean']:.4f} (all pairs)")
            
            print("\n  => Structural (LPIPS) diversity:")
            lpips_device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_built() else "cpu")
            real_lpips_div = _compute_lpips_diversity(real_paths, n_bootstrap=150, device=lpips_device)
            synth_lpips_div = _compute_lpips_diversity(synth_paths, n_bootstrap=150, device=lpips_device)
            print(f"     Real   : {real_lpips_div['mean']:.4f}  (95% CI: {real_lpips_div['ci_lower']:.4f} - {real_lpips_div['ci_upper']:.4f})")
            print(f"     Synth  : {synth_lpips_div['mean']:.4f}  (95% CI: {synth_lpips_div['ci_lower']:.4f} - {synth_lpips_div['ci_upper']:.4f})")
            
            # -- Source Fidelity Analysis (Conditioning Strength) ---------------------
            metrics_path = out_dir / "metrics_summary.json"

            if do_synth and synth_paths and not args.no_prompt_level:
                _, json_samples = _load_prompt_json(prompt_json_path)
                paired_samples = _pair_synthetic_with_samples(synth_paths, json_samples)
                
                if paired_samples:
                    print("\n[analyse] computing conditioning fidelity (source vs output) …")
                    
            # Map source (real) image paths
            source_paths = []
            for s in paired_samples:
                img_path = s.get("controlnet_image") or s.get("ip_image")
                source_paths.append(Path(img_path) if img_path else None)
            
            # Filter pairs where source image exists
            valid_synth_paths, valid_source_paths, valid_synth_embeds = [], [], []
            for i, src_p in enumerate(source_paths):
                if src_p and src_p.exists() and i < len(synth_embeds):
                    valid_synth_paths.append(synth_paths[i])
                    valid_source_paths.append(src_p)
                    valid_synth_embeds.append(synth_embeds[i])
            if valid_synth_paths:
                print(f"   => extracting source CLIP embeddings for {len(valid_source_paths)} images …")
                # 1. Pixel space
                pixel_fid = _compute_conditioning_fidelity(valid_synth_paths, valid_source_paths)
                
                # 2. CLIP space
                source_embeds_raw, _ = extract_clip_embeddings(valid_source_paths, batch_size=args.batch_size)
                source_embeds = source_embeds_raw / np.maximum(np.linalg.norm(source_embeds_raw, axis=1, keepdims=True), 1e-8)
                clip_fid = _compute_clip_fidelity(np.stack(valid_synth_embeds), source_embeds)
                
                print(f"   => Mean Fidelity: Pixel={pixel_fid['mean_dist']:.4f}, CLIP={clip_fid['mean_dist']:.4f}")
                
                # Plot absolute-distance fidelity (empirical percentile bands)
                _save_fidelity_plot(pixel_fid["distances"], clip_fid["distances"], class_name, out_dir / f"conditioning_fidelity_{suffix}.png")

                # 3. Memorization Ratio R = d(syn, src) / d(syn, distractor)
                print("\n[analyse] computing memorization ratio R (Yoon-adapted) …")

                # Pixel-space ratio (uses a pool of training images as distractors)
                print("   => Pixel-space ratio …")
                pixel_ratio = _compute_memorization_ratio_pixel(
                    valid_synth_paths, valid_source_paths, real_paths,
                )

                # CLIP-space ratio (uses full training embedding matrix as distractors)
                print("   => CLIP-space ratio …")
                valid_synth_embeds_arr = np.stack(valid_synth_embeds)
                clip_ratio = _compute_memorization_ratio_clip(
                    valid_synth_embeds_arr, source_embeds, train_embeds,
                )

                print(f"   => Memorization Ratio R:")
                print(f"      Pixel  — mean={pixel_ratio['mean_ratio']:.4f}  median={pixel_ratio['median_ratio']:.4f}  (n={len(pixel_ratio['ratios'])})")
                print(f"      CLIP   — mean={clip_ratio['mean_ratio']:.4f}  median={clip_ratio['median_ratio']:.4f}  (n={len(clip_ratio['ratios'])})")

                # Plot ratio histograms (using calibrated τ if available)
                _save_memorization_ratio_plot(
                    pixel_ratio["ratios"], clip_ratio["ratios"],
                    class_name, out_dir / f"memorization_ratio_{suffix}.png",
                )
                
                if metrics_path.exists():
                    with metrics_path.open("r", encoding="utf-8") as f:
                        metrics_data = json.load(f)
                    if "synthetic" not in metrics_data: metrics_data["synthetic"] = {}
                    metrics_data["synthetic"]["conditioning_fidelity"] = {
                        "pixel": {k: v for k, v in pixel_fid.items() if k != "distances"},
                        "clip": {k: v for k, v in clip_fid.items() if k != "distances"}
                    }
                    metrics_data["synthetic"]["memorization_ratio"] = {
                        "pixel": {k: v for k, v in pixel_ratio.items() if k != "ratios"},
                        "clip": {k: v for k, v in clip_ratio.items() if k != "ratios"},
                    }
                    with metrics_path.open("w", encoding="utf-8") as f:
                        json.dump(metrics_data, f, indent=2)
                    
                    with (out_dir / f"fidelity_distances_pixel_{suffix}.json").open("w") as f:
                        json.dump(pixel_fid["distances"], f)
                    with (out_dir / f"fidelity_distances_clip_{suffix}.json").open("w") as f:
                        json.dump(clip_fid["distances"], f)
                    with (out_dir / f"memorization_ratios_pixel_{suffix}.json").open("w") as f:
                        json.dump(pixel_ratio["ratios"], f)
                    with (out_dir / f"memorization_ratios_clip_{suffix}.json").open("w") as f:
                        json.dump(clip_ratio["ratios"], f)
            else:
                print("   [analyse] skipping fidelity analysis: no valid source images found on disk.")

            # Append to the previously generated metrics file
            if metrics_path.exists():
                with metrics_path.open("r", encoding="utf-8") as f:
                    metrics_data = json.load(f)
                
                if "real" not in metrics_data: metrics_data["real"] = {}
                metrics_data["real"]["semantic_diversity_clip"] = real_div
                metrics_data["real"]["structural_diversity_lpips"] = real_lpips_div
                
                if "synthetic" not in metrics_data: metrics_data["synthetic"] = {}
                metrics_data["synthetic"]["semantic_diversity_clip"] = synth_div
                metrics_data["synthetic"]["structural_diversity_lpips"] = synth_lpips_div
                
                metrics_data["real"].pop("pairwise_diversity", None)
                metrics_data["synthetic"].pop("pairwise_diversity", None)
                
                with metrics_path.open("w", encoding="utf-8") as f:
                    json.dump(metrics_data, f, indent=2)
                print(f"\n[analyse] appended diversity metrics -> {metrics_path.name}")

        # UMAP projection (all points fit together for consistent mapping)
        all_embeds = np.concatenate(
            [train_embeds, val_embeds, synth_embeds], axis=0,
        )
        embedding_2d = _run_umap(
            all_embeds,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
        )

        n_train = len(train_embeds)
        n_val = len(val_embeds)
        train_2d = embedding_2d[:n_train]
        val_2d = embedding_2d[n_train : n_train + n_val]
        synth_2d = embedding_2d[n_train + n_val :]

        # Compute cluster metrics
        umap_metrics = _compute_umap_metrics(synth_2d, train_2d)

        # Save UMAP outputs
        dataset_label = prompt_json_path.stem if prompt_json_path else "synthetic"

        _save_umap_plot(
            train_2d, val_2d, synth_2d,
            dataset_label, umap_metrics,
            class_name, out_dir / f"umap_clip_{suffix}.png",
        )
        _save_umap_metrics(
            umap_metrics, class_name, out_dir / f"umap_metrics_{suffix}.json",
        )

    print(f"\n[analyse] all outputs written to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
