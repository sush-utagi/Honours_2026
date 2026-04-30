#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# run_all_evaluations.sh
#
# Evaluates every trained model (baseline A/B, experimental A/B)
# on both the validation and test splits, saving reports and
# confusion matrices into neatly separated subdirectories under
# results/.
#
# Directory layout produced:
#   results/
#   ├── baseline_model_A/
#   │   ├── val/
#   │   │   ├── <timestamp>_baseline_report.txt
#   │   │   └── <timestamp>_baseline_val_cm/
#   │   └── test/
#   │       ├── <timestamp>_baseline_report.txt
#   │       └── <timestamp>_baseline_test_cm/
#   ├── baseline_model_B/
#   │   ├── val/ ...
#   │   └── test/ ...
#   ├── experimental_model_A/
#   │   ├── val/ ...
#   │   └── test/ ...
#   └── experimental_model_B/
#       ├── val/ ...
#       └── test/ ...
#
# Usage:
#   bash scripts/run_all_evaluations.sh
# ─────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

EVAL_SCRIPT="$PROJECT_ROOT/experiments/model/test_model.py"
RUNS_DIR="$PROJECT_ROOT/runs"
RESULTS_DIR="$PROJECT_ROOT/results"

# Each entry is "model_dir_name:model_type_flag"
MODELS="baseline_model_A:baseline
baseline_model_B:baseline_B
experimental_model_A:experimental
experimental_model_B:experimental_B"

SPLITS="val test"

echo "============================================================"
echo "  Running evaluations for all models on val + test splits"
echo "============================================================"
echo ""

for entry in $MODELS; do
    model_name="${entry%%:*}"
    model_type="${entry##*:}"
    ckpt="$RUNS_DIR/$model_name/checkpoints/best.pt"

    if [ ! -f "$ckpt" ]; then
        echo "[SKIP] $model_name — checkpoint not found: $ckpt"
        echo ""
        continue
    fi

    for split in $SPLITS; do
        out_dir="$RESULTS_DIR/$model_name/$split"
        mkdir -p "$out_dir"

        echo "────────────────────────────────────────────────────────────"
        echo "  Model:  $model_name"
        echo "  Split:  $split"
        echo "  Ckpt:   $ckpt"
        echo "  Output: $out_dir"
        echo "────────────────────────────────────────────────────────────"

        PYTORCH_ENABLE_MPS_FALLBACK=1 python3 "$EVAL_SCRIPT" \
            --model-type "$model_type" \
            --ckpt "$ckpt" \
            --split "$split" \
            --results-dir "$out_dir" \
            --plot-confusion

        echo ""
        echo "[DONE] $model_name / $split"
        echo ""
    done
done

echo "============================================================"
echo "  All evaluations complete. Results saved under:"
echo "  $RESULTS_DIR"
echo "============================================================"
