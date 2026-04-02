#!/bin/bash
# ============================================================
# Sync checkpoints from Wulver to local
# ============================================================
# Downloads the latest checkpoints (main + ablation models).
# Requires active SSH multiplexing: run `ssh wulver` first.
#
# Usage:
#   bash scripts/wulver_sync_model.sh
# ============================================================

set -e
cd ~/cs785-project

WULVER_DIR="/course/2026/spring/cs/785/hz79/adp232/cs785"
LOCAL_DIR="checkpoints"

echo "Syncing checkpoints from Wulver..."
echo ""

# Main model
echo -n "  best_model.pt ... "
scp -q wulver:$WULVER_DIR/checkpoints/best_model.pt $LOCAL_DIR/best_model.pt 2>/dev/null && \
    echo "OK ($(du -h $LOCAL_DIR/best_model.pt | cut -f1))" || echo "SKIP (not found or SSH unavailable)"

# Ablation models
for m in ablation_model2 ablation_model3 ablation_model4; do
    mkdir -p "$LOCAL_DIR/$m"
    echo -n "  $m/best_model.pt ... "
    scp -q wulver:$WULVER_DIR/checkpoints/$m/best_model.pt $LOCAL_DIR/$m/best_model.pt 2>/dev/null && \
        echo "OK ($(du -h $LOCAL_DIR/$m/best_model.pt | cut -f1))" || echo "SKIP (not found)"
done

# Pretrained encoder
echo -n "  pretrained_encoder.pt ... "
scp -q wulver:$WULVER_DIR/checkpoints/pretrained_encoder.pt $LOCAL_DIR/pretrained_encoder.pt 2>/dev/null && \
    echo "OK ($(du -h $LOCAL_DIR/pretrained_encoder.pt | cut -f1))" || echo "SKIP (not found)"

echo ""
echo "Local checkpoints:"
ls -lh $LOCAL_DIR/*.pt $LOCAL_DIR/*/best_model.pt 2>/dev/null | awk '{print "  " $NF " (" $5 ")"}'
echo ""
echo "Done!"
