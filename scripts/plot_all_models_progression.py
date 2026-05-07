#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt
from pathlib import Path

# Config
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "results" / "epoch_progression"
FIG_DIR = PROJECT_ROOT / "experiments" / "figures"
OUT_PATH = FIG_DIR / "all_models_f1_progression.png"

MODELS = {
    "baseline_model_A": {"label": "Baseline (Real Only)", "color": "#2196F3", "marker": "o", "alpha": 0.4},
    "baseline_model_B": {"label": "Baseline (Classical Aug)", "color": "#00ACC1", "marker": "s", "alpha": 0.6},
    "experimental_model_A": {"label": "Experimental (IP-Adapter)", "color": "#E53935", "marker": "^", "alpha": 1.0},
    "experimental_model_B": {"label": "Experimental (ControlNet)", "color": "#FF9800", "marker": "D", "alpha": 1.0},
}

REPORTS = {
    "baseline_model_A": "baseline_model_A/test/20260426_151826_baseline_report.txt",
    "baseline_model_B": "baseline_model_B/test/20260426_153206_baseline_B_report.txt",
    "experimental_model_A": "experimental_model_A/test/20260426_154539_experimental_report.txt",
    "experimental_model_B": "experimental_model_B/test/20260426_155902_experimental_B_report.txt",
}

CLASSES = ["toaster", "hair drier"]

def parse_report_final_metrics(path: Path):
    if not path.exists(): return None, None
    acc, macro_f1 = None, None
    with open(path, "r") as f:
        for line in f:
            if line.startswith("Accuracy:"): acc = float(line.split(":")[1].strip())
            if line.startswith("Macro F1:"): macro_f1 = float(line.split(":")[1].strip())
    return acc, macro_f1

def main():
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    
    print("\n" + "="*60)
    print(f"{'Model Name':<30} | {'Accuracy':<10} | {'Macro F1':<10}")
    print("-" * 60)

    for model_id, config in MODELS.items():
        # 1. Load Data
        json_path = RAW_DIR / f"{model_id}_test.json"
        if not json_path.exists():
            print(f"Warning: {json_path.name} not found. Skipping.")
            continue
            
        with open(json_path, "r") as f:
            data = json.load(f)
        
        epochs = sorted([int(e) for e in data.keys()])
        
        # 2. Plot for each class
        for i, cls_name in enumerate(CLASSES):
            f1_scores = [data[str(e)][cls_name]["f1"] for e in epochs]
            axes[i].plot(epochs, f1_scores, 
                         marker=config["marker"], 
                         color=config["color"], 
                         alpha=config["alpha"],
                         linewidth=3, 
                         markersize=9, 
                         label=config["label"])
        
        # 3. Print Final Report Metrics
        report_path = PROJECT_ROOT / "results" / REPORTS[model_id]
        acc, macro_f1 = parse_report_final_metrics(report_path)
        print(f"{config['label']:<30} | {acc if acc else 'N/A':<10.4f} | {macro_f1 if macro_f1 else 'N/A':<10.4f}")

    # 4. Styling
    for i, cls_name in enumerate(CLASSES):
        axes[i].set_title(f"{cls_name.title()}", fontsize=18, fontweight='bold', pad=15)
        axes[i].set_xlabel("Epoch", fontsize=14)
        axes[i].set_ylim(-0.02, 0.6)
        axes[i].grid(True, alpha=0.3)
        if i == 0:
            axes[i].set_ylabel("F1 Score (Test Set)", fontsize=14)
        if i == 1:
            axes[i].legend(fontsize=11, loc="upper right", framealpha=0.9)

    fig.suptitle("Minority-Class F1 Progression (Test Set)", fontsize=22, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print("\n" + "="*60)
    print(f"[plot] Combined comparison plot saved to: {OUT_PATH}")

if __name__ == "__main__":
    main()
