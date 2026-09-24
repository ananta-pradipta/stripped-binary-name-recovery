#!/bin/bash
# Sync code changes to HPC (excludes large data/checkpoints)
# Uses rsync — only transfers files that changed, takes seconds.
#
# Requires SSH multiplexing (ssh HPC in another terminal first)
#
# Usage: bash scripts/HPC_sync.sh

HPC="HPC"
REMOTE_DIR="/course/2026/spring/cs/785/ACCOUNT/USER/cs785"
LOCAL_DIR="$HOME/cs785-project"

echo "Syncing code to HPC..."
rsync -avz --delete \
  --exclude 'data/' \
  --exclude 'checkpoints/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.git/' \
  --exclude 'cs785-env/' \
  --exclude 'demo/stripped/' \
  --exclude 'demo/raw/' \
  --exclude 'results/exp*' \
  ${LOCAL_DIR}/src/ ${HPC}:${REMOTE_DIR}/src/
rsync -avz \
  ${LOCAL_DIR}/configs/ ${HPC}:${REMOTE_DIR}/configs/
rsync -avz \
  --exclude '__pycache__/' \
  ${LOCAL_DIR}/scripts/ ${HPC}:${REMOTE_DIR}/scripts/

echo "Done! Synced: src/, configs/, scripts/"
