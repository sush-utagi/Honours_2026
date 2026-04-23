#!/usr/bin/env bash

# exi on any error
set -e

echo "  Starting Analysis Pipeline Evaluation "

echo ""
echo "[1/4] Analyzing Toaster (IP-Adapter)..."
python -m experiments.evaluation_module.grade_images.analyse \
    --class-name toaster --synthetic-dir data_generation_outputs/toaster_600_ip

echo ""
echo "[2/4] Analyzing Toaster (ControlNet)..."
python -m experiments.evaluation_module.grade_images.analyse \
    --class-name toaster --synthetic-dir data_generation_outputs/toaster_600_cn

echo ""
echo "[3/4] Analyzing Hair Drier (IP-Adapter)..."
python -m experiments.evaluation_module.grade_images.analyse \
    --class-name "hair drier" --synthetic-dir data_generation_outputs/hair_drier_600_ip

echo ""
echo "[4/4] Analyzing Hair Drier (ControlNet)..."
python -m experiments.evaluation_module.grade_images.analyse \
    --class-name "hair drier" --synthetic-dir data_generation_outputs/hair_drier_600_cn

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
