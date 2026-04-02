#!/bin/bash
# ============================================================
# Submit evaluation job to Wulver
# ============================================================
# Usage:
#   bash scripts/evaluate_wulver.sh          # Sync code + submit full eval
#   bash scripts/evaluate_wulver.sh status   # Check job status
#   bash scripts/evaluate_wulver.sh log      # Show latest eval output
# ============================================================

set -e
cd ~/cs785-project

WULVER_DIR="/course/2026/spring/cs/785/hz79/adp232/cs785"

case "${1:-submit}" in
    submit)
        echo "Syncing code to Wulver..."
        bash scripts/wulver_sync.sh

        echo ""
        echo "Submitting evaluation job..."
        ssh wulver "cd $WULVER_DIR && sbatch scripts/wulver_eval_full.sbatch"

        echo ""
        echo "Use 'bash scripts/evaluate_wulver.sh status' to check progress"
        echo "Use 'bash scripts/evaluate_wulver.sh log' to view output"
        ;;
    status)
        ssh wulver "squeue -u adp232"
        ;;
    log)
        # Find the latest eval output file
        LATEST=$(ssh wulver "ls -t $WULVER_DIR/cs785-eval.*.out 2>/dev/null | head -1")
        if [ -z "$LATEST" ]; then
            echo "No evaluation output files found."
            exit 1
        fi
        echo "Latest output: $LATEST"
        echo "================================================================"
        ssh wulver "sed 's/\r/\n/g' $LATEST | grep -v 'it/s' | grep -v '^\$'"
        ;;
    *)
        echo "Usage: bash scripts/evaluate_wulver.sh [submit|status|log]"
        exit 1
        ;;
esac
