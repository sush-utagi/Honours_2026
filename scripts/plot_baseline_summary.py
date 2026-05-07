#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = PROJECT_ROOT / "results" / "epoch_progression" / "baseline_model_A_test.json"
REPORT_PATH = PROJECT_ROOT / "results" / "baseline_model_A" / "test" / "20260426_151826_baseline_report.txt"
OUT_PATH = PROJECT_ROOT / "experiments" / "figures" / "baseline_minority_f1.png"

def parse_report_final_metrics(path: Path):
    """Extract Accuracy and Macro F1 from the final report text file."""
    acc = None
    macro_f1 = None
    with open(path, "r") as f:
        for line in f:
            if line.startswith("Accuracy:"):
                acc = float(line.split(":")[1].strip())
            if line.startswith("Macro F1:"):
                macro_f1 = float(line.split(":")[1].strip())
    return acc, macro_f1

def main():
    # 1. Load per-epoch class data
    if not JSON_PATH.exists():
        print(f"Error: {JSON_PATH} not found.")
        return
    
    with open(JSON_PATH, "r") as f:
        data = json.load(f)
    
    # JSON keys are stringified epoch numbers
    epochs = sorted([int(e) for e in data.keys()])
    toaster_f1 = [data[str(e)]["toaster"]["f1"] for e in epochs]
    hair_dryer_f1 = [data[str(e)]["hair drier"]["f1"] for e in epochs]

    # 2. Extract final whole model performance
    acc, macro_f1 = parse_report_final_metrics(REPORT_PATH)

    # 3. Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, toaster_f1, 'o-', color='#FF9800', linewidth=3, markersize=8, label="Toaster (F1)")
    plt.plot(epochs, hair_dryer_f1, 's-', color='#E53935', linewidth=3, markersize=8, label="Hair Dryer (F1)")
    
    # Horizontal line for final macro F1 if we want to show context, but user just asked to plot classes.
    # I'll just print the global metrics as requested.

    plt.title("Baseline Model: Minority-Class Performance Progression (Test Set)", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("F1 Score", fontsize=14)
    plt.ylim(-0.02, 0.5)  # Zoomed into the minority class range
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=200)
    print(f"\n[plot] Baseline minority F1 plot saved to: {OUT_PATH}")
    
    # 4. Print results
    print("\n" + "="*40)
    print("FINAL BASELINE PERFORMANCE (WHOLE MODEL)")
    print("="*40)
    print(f"Overall Accuracy: {acc:.4f}")
    print(f"Macro F1 Score:   {macro_f1:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
