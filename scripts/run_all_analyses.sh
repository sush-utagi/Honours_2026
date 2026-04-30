#!/usr/bin/env bash

# exi on any error
set -e

echo "  Starting Analysis Pipeline Evaluation "

echo ""
echo "[1/4] Analyzing Toaster (IP-Adapter)..."
python -m experiments.evaluation_module.grade_images.analyse \
    --class-name toaster --synthetic-dir data_generation_outputs/diffusion_based_augmentations_ip/toaster

echo ""
echo "[2/4] Analyzing Toaster (ControlNet)..."
python -m experiments.evaluation_module.grade_images.analyse \
    --class-name toaster --synthetic-dir data_generation_outputs/diffusion_based_augmentation_cn/toaster

echo ""
echo "[3/4] Analyzing Hair Drier (IP-Adapter)..."
python -m experiments.evaluation_module.grade_images.analyse \
    --class-name "hair drier" --synthetic-dir "data_generation_outputs/diffusion_based_augmentations_ip/hair drier"

echo ""
echo "[4/4] Analyzing Hair Drier (ControlNet)..."
python -m experiments.evaluation_module.grade_images.analyse \
    --class-name "hair drier" --synthetic-dir data_generation_outputs/diffusion_based_augmentation_cn/hair_drier

echo ""
echo "  Generating Consolidated Plots "

echo "=> Plotting New Diversity Histograms..."
python experiments/evaluation_module/grade_images/plot_diversities.py

echo "=> Plotting Domain Scores (Mean, Medians)..."
python experiments/evaluation_module/grade_images/plot_domain_scores.py

echo "=> Plotting Global Distances (Frechet, Wasserstein)..."
python experiments/evaluation_module/grade_images/plot_distances.py

echo ""
echo "  Pipeline Complete! Results in results/clip_analysis/ "
