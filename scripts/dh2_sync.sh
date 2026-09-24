#!/bin/bash
# Sync the dualhead-hydra (dataset v2) workspace to Wulver — ADDITIVE, isolated under dh2/.
# Usage: bash scripts/dh2_sync.sh [code|data|all]   (default: code)
set -euo pipefail
W=wulver
R=/project/hz79/_shared/cs785/dh2
L=$HOME/cs785-project
what=${1:-code}
ssh $W "mkdir -p $R/src $R/scripts $R/configs $R/tests $R/docs $R/data $R/checkpoints $R/results $R/slurm"
if [[ $what == code || $what == all ]]; then
  rsync -az --delete --exclude '__pycache__/' --exclude '*.pyc' $L/src/ $W:$R/src/
  rsync -az --exclude '__pycache__/' $L/scripts/ $W:$R/scripts/
  rsync -az $L/configs/ $W:$R/configs/
  rsync -az --exclude '__pycache__/' $L/tests/ $W:$R/tests/
  rsync -az $L/docs/ $W:$R/docs/
  rsync -az $L/activate.sh $L/requirements.txt $W:$R/ 2>/dev/null || true
  echo "code synced -> $W:$R (git $(git -C $L rev-parse --short HEAD))"
fi
if [[ $what == data || $what == all ]]; then
  # dataset v2 artefacts only (graphs_v3 is the big one: one dir per binary + index files)
  for d in graphs_v3 string_refs_v2 labels_v2; do
    rsync -az --info=progress2 $L/data/$d/ $W:$R/data/$d/
  done
  rsync -az $L/data/match_index_v2.json $L/data/match_index_v2_report.tsv $L/data/split_v2.json $L/data/votes_vocab_v2.json $W:$R/data/ 2>/dev/null || true
  echo "data synced"
fi
