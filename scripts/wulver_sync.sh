#!/bin/bash
# Sync code changes to HPC (excludes large data/checkpoints)
# Uses rsync — only transfers files that changed, takes seconds.
#
# Requires SSH multiplexing (ssh <hpc-host> in another terminal first)
#
# Usage: bash scripts/wulver_sync.sh

WULVER="<hpc-host>"
REMOTE_DIR="<project-root>"
LOCAL_DIR="$HOME/bfnr-project"

echo "Syncing code to HPC..."
rsync -avz --delete \
  --exclude 'data/' \
  --exclude 'checkpoints/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.git/' \
  --exclude 'bfnr-env/' \
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
