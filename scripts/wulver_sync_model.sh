#!/bin/bash
# ============================================================
# Sync checkpoints from HPC to local
# ============================================================
# Downloads the latest checkpoints (main + ablation models).
# Requires active SSH multiplexing: run `ssh <hpc-host>` first.
#
# Usage:
#   bash scripts/wulver_sync_model.sh
# ============================================================

set -e
cd ~/bfnr-project

WULVER_DIR="<project-root>"
LOCAL_DIR="checkpoints"

echo "Syncing checkpoints from HPC..."
echo ""

# Main model
echo -n "  best_model.pt ... "
scp -q <hpc-host>:$WULVER_DIR/checkpoints/best_model.pt $LOCAL_DIR/best_model.pt 2>/dev/null && \
    echo "OK ($(du -h $LOCAL_DIR/best_model.pt | cut -f1))" || echo "SKIP (not found or SSH unavailable)"

# Ablation models
for m in ablation_model2 ablation_model3 ablation_model4; do
    mkdir -p "$LOCAL_DIR/$m"
    echo -n "  $m/best_model.pt ... "
    scp -q <hpc-host>:$WULVER_DIR/checkpoints/$m/best_model.pt $LOCAL_DIR/$m/best_model.pt 2>/dev/null && \
        echo "OK ($(du -h $LOCAL_DIR/$m/best_model.pt | cut -f1))" || echo "SKIP (not found)"
done

# Pretrained encoder
echo -n "  pretrained_encoder.pt ... "
scp -q <hpc-host>:$WULVER_DIR/checkpoints/pretrained_encoder.pt $LOCAL_DIR/pretrained_encoder.pt 2>/dev/null && \
    echo "OK ($(du -h $LOCAL_DIR/pretrained_encoder.pt | cut -f1))" || echo "SKIP (not found)"

echo ""
echo "Local checkpoints:"
ls -lh $LOCAL_DIR/*.pt $LOCAL_DIR/*/best_model.pt 2>/dev/null | awk '{print "  " $NF " (" $5 ")"}'
echo ""
echo "Done!"
