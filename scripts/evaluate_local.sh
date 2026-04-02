#!/bin/bash
# ============================================================
# Local Evaluation — Run prediction + evaluation locally
# ============================================================
# k-NN hybrid is the default inference method.
#
# Usage:
#   bash scripts/evaluate_local.sh              # Full report (test+demo+samples+ablation)
#   bash scripts/evaluate_local.sh test         # Test/val metrics only
#   bash scripts/evaluate_local.sh demo         # Demo on unseen packages only
#   bash scripts/evaluate_local.sh ablation     # Ablation table only
#   bash scripts/evaluate_local.sh samples      # Test + demo with sample predictions
#   bash scripts/evaluate_local.sh decoder-only # Same as 'all' but without k-NN
#
# Options:
#   CHECKPOINT=path/to/model.pt bash scripts/evaluate_local.sh test
# ============================================================

set -e
source ~/cs785-project/activate.sh
cd ~/cs785-project

MODE="${1:-all}"
CHECKPOINT="${CHECKPOINT:-checkpoints/best_model.pt}"
CONFIG="configs/optimized_large.yaml"

if [ ! -f "$CHECKPOINT" ]; then
    if [ -f "checkpoints/best_model_wulver.pt" ]; then
        CHECKPOINT="checkpoints/best_model_wulver.pt"
    else
        echo "  Checkpoint not found: $CHECKPOINT"
        echo "  Run: bash scripts/wulver_sync_model.sh"
        exit 1
    fi
fi

case "$MODE" in
    test)
        python3 scripts/eval_full.py --checkpoint "$CHECKPOINT" --config "$CONFIG" \
            --sections info test
        ;;
    demo)
        python3 scripts/eval_full.py --checkpoint "$CHECKPOINT" --config "$CONFIG" \
            --sections info demo
        ;;
    ablation)
        python3 scripts/eval_full.py --checkpoint "$CHECKPOINT" --config "$CONFIG" \
            --sections info ablation
        ;;
    samples)
        python3 scripts/eval_full.py --checkpoint "$CHECKPOINT" --config "$CONFIG" \
            --sections info test demo samples
        ;;
    decoder-only)
        python3 scripts/eval_full.py --checkpoint "$CHECKPOINT" --config "$CONFIG" \
            --sections info test demo samples ablation --no-knn
        ;;
    all)
        python3 scripts/eval_full.py --checkpoint "$CHECKPOINT" --config "$CONFIG" \
            --sections info test demo samples ablation
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: bash scripts/evaluate_local.sh [test|demo|ablation|samples|decoder-only|all]"
        exit 1
        ;;
esac
