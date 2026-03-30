#!/bin/bash
# Sync code changes to Wulver (excludes large data/checkpoints)
# Uses rsync — only transfers files that changed, takes seconds.
#
# Requires SSH multiplexing (ssh wulver in another terminal first)
#
# Usage: bash scripts/wulver_sync.sh

WULVER="wulver"
REMOTE_DIR="/course/2026/spring/cs/785/hz79/adp232/cs785"
LOCAL_DIR="$HOME/cs785-project"

echo "Syncing code to Wulver..."
rsync -avz --delete \
  --exclude 'data/' \
  --exclude 'checkpoints/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.git/' \
  --exclude '.claude/' \
  --exclude 'cs785-env/' \
  --exclude 'demo/stripped/' \
  --exclude 'demo/raw/' \
  --exclude 'results/exp*' \
  ${LOCAL_DIR}/src/ ${WULVER}:${REMOTE_DIR}/src/
rsync -avz \
  ${LOCAL_DIR}/configs/ ${WULVER}:${REMOTE_DIR}/configs/
rsync -avz \
  --exclude '__pycache__/' \
  ${LOCAL_DIR}/scripts/ ${WULVER}:${REMOTE_DIR}/scripts/

echo "Done! Synced: src/, configs/, scripts/"
