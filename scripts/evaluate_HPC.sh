#!/bin/bash
# ============================================================
# Submit evaluation job to HPC
# ============================================================
# Usage:
#   bash scripts/evaluate_HPC.sh          # Sync code + submit full eval
#   bash scripts/evaluate_HPC.sh status   # Check job status
#   bash scripts/evaluate_HPC.sh log      # Show latest eval output
# ============================================================

set -e
cd ~/cs785-project

HPC_DIR="/course/2026/spring/cs/785/ACCOUNT/USER/cs785"

case "${1:-submit}" in
    submit)
        echo "Syncing code to HPC..."
        bash scripts/HPC_sync.sh

        echo ""
        echo "Submitting evaluation job..."
        ssh HPC "cd $HPC_DIR && sbatch scripts/HPC_eval_full.sbatch"

        echo ""
        echo "Use 'bash scripts/evaluate_HPC.sh status' to check progress"
        echo "Use 'bash scripts/evaluate_HPC.sh log' to view output"
        ;;
    status)
        ssh HPC "squeue -u USER"
        ;;
    log)
        # Find the latest eval output file
        LATEST=$(ssh HPC "ls -t $HPC_DIR/cs785-eval.*.out 2>/dev/null | head -1")
        if [ -z "$LATEST" ]; then
            echo "No evaluation output files found."
            exit 1
        fi
        echo "Latest output: $LATEST"
        echo "================================================================"
        ssh HPC "sed 's/\r/\n/g' $LATEST | grep -v 'it/s' | grep -v '^\$'"
        ;;
    *)
        echo "Usage: bash scripts/evaluate_HPC.sh [submit|status|log]"
        exit 1
        ;;
esac
