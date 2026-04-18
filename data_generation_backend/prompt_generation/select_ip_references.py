#!/usr/bin/env python3
"""
Select representative reference images for a given directory using
K-Means clustering on CLIP embeddings.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# Add root so we can import grader and dataset assembler
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.cluster import KMeans

# We reuse the robust CLIP evaluator we already built!
from experiments.evaluation_module.grade_images.grader import extract_clip_embeddings
from experiments.dataset_creation.dataset_assembler import HybridDatasetAssembler

def main():
    parser = argparse.ArgumentParser(description="Select reference images using CLIP + KMeans.")
    parser.add_argument("--image_dir", type=str, default=str(ROOT / "coco_dataset" / "contextual_crops"), help="Path to images (or contextual_crops root).")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory for selected images.")
    parser.add_argument("--classes", type=str, nargs="+", default=["toaster", "hair drier"], help="Optional: filter by class (e.g. 'toaster' 'hair drier') if image_dir is contextual_crops root.")
    parser.add_argument("--k", type=int, default=7, help="Number of clusters (selected images) per group.")
    parser.add_argument("--visualise", action="store_true", help="Generate a PCA scatter plot of the clusters.")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    paths_by_class = {}
    
    if args.classes:
        print(f"Filtering {image_dir} for classes: {args.classes} using Dataset Assembler...")
        assembler = HybridDatasetAssembler(contextual_root=str(image_dir))
        paths_by_class = assembler.get_image_paths_by_class(split="train", target_classes=args.classes)
        for cls_name, paths in paths_by_class.items():
            print(f"Assembler yielded {len(paths)} matching images for '{cls_name}'.")
    else:
        # Fallback to standard globbing if no classes are specified
        paths = [p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
        paths_by_class["dataset"] = paths
        print(f"Found {len(paths)} images via raw directory search.")

    if not any(paths_by_class.values()):
        print("No images found! Check your directory or class names.")
        return

    base_output_dir = Path(args.output_dir) / "selected_references"

    for cls_name, image_paths in paths_by_class.items():
        if not image_paths:
            continue
            
        print(f"\n{'='*50}")
        print(f"Processing class: {cls_name} ({len(image_paths)} images)")
        print(f"{'='*50}")

        # 2. Extract CLIP embeddings (these are L2 normalized by grader!)
        print("Extracting CLIP embeddings...")
        embeddings, valid_indices = extract_clip_embeddings(image_paths, batch_size=32)
        
        if len(valid_indices) < args.k:
            print(f"Not enough valid images ({len(valid_indices)}) to form {args.k} clusters. Skipping.")
            continue

        # Map back successfully encoded embeddings to their original paths
        valid_paths = [image_paths[i] for i in valid_indices]

        # 3. KMeans clustering
        print(f"Running KMeans with K={args.k}...")
        kmeans = KMeans(n_clusters=args.k, random_state=42, n_init="auto")
        kmeans.fit(embeddings)
        centroids = kmeans.cluster_centers_  # shape: (K, 768)
        cluster_sizes = np.bincount(kmeans.labels_, minlength=args.k)
        
        # L2 normalize centroids to compute cosine similarity correctly
        centroids_norm = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)

        # 4. Optional: Visualise Clusters
        # Output paths for this class
        # Spaces in class name replaced by underscore for safety
        safe_cls_name = cls_name.replace(" ", "_")
        cls_output_dir = base_output_dir / safe_cls_name
        cls_output_dir.mkdir(parents=True, exist_ok=True)

        if args.visualise:
            from sklearn.decomposition import PCA
            import matplotlib.pyplot as plt
            
            print("Generating cluster visualization (PCA)...")
            pca = PCA(n_components=2)
            reduced = pca.fit_transform(embeddings)
            
            plt.figure(figsize=(10, 8))
            # Define colors for clusters
            scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=kmeans.labels_, cmap='tab10', alpha=0.6, s=15)
            plt.colorbar(scatter, label='Cluster Index')
            
            # Project centroids as well
            centroids_2d = pca.transform(centroids)
            plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='red', marker='x', s=100, label='Centroids')
            
            plt.title(f"K-Means Clusters (K={args.k}) for {cls_name}\nInertia: {kmeans.inertia_:.2f}")
            plt.xlabel("PCA 1")
            plt.ylabel("PCA 2")
            plt.legend()
            
            viz_path = cls_output_dir / "cluster_visualization.png"
            plt.savefig(viz_path, dpi=150)
            plt.close()
            print(f"↳ Saved cluster visualization to {viz_path}")

        # 5. Find highest cosine similarity per cluster
        print("Finding best representative images...")
        similarities = np.dot(embeddings, centroids_norm.T)
        best_indices = np.argmax(similarities, axis=0)

        metadata = []

        # 5 & 6. Copy files and log metadata
        for cluster_idx in range(args.k):
            best_img_idx = best_indices[cluster_idx]
            best_sim = float(similarities[best_img_idx, cluster_idx])
            src_path = valid_paths[best_img_idx]
            
            dst_path = cls_output_dir / src_path.name
            shutil.copy2(src_path, dst_path)
            
            metadata.append({
                "filename": src_path.name,
                "cluster_index": cluster_idx,
                "cluster_size": int(cluster_sizes[cluster_idx]),
                "cosine_similarity": round(best_sim, 5),
                "original_path": str(src_path)
            })
            print(f"Cluster {cluster_idx+1}/{args.k}: Selected {src_path.name} (Sim: {best_sim:.4f})")

        meta_path = cls_output_dir / "metadata.json"
        
        output_metadata = {
            "total_cluster_size": int(np.sum(cluster_sizes)),
            "references": metadata
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(output_metadata, f, indent=4)
            
        print(f"↳ Saved {args.k} images to {cls_output_dir}")

    print(f"\nAll operations complete! Outputs located in: {base_output_dir}")

if __name__ == "__main__":
    main()
