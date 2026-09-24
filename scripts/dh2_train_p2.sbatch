#!/bin/bash
#SBATCH --job-name=cs785-eval --account=hz79 --partition=gpu --qos=standard
#SBATCH --gres=gpu:a100:1 --cpus-per-task=8 --mem=120G --time=24:00:00
#SBATCH --output=/project/hz79/_shared/cs785/dh2/slurm/p2_%j.out
# P2: CCS architecture unchanged (configs/dualhead_v2_large.yaml), dataset v2, split policy v3,
# model selection on val_xproj. Honest baseline for the dual-head ladder.
set -uo pipefail
cd /project/hz79/_shared/cs785/dh2
module load CUDA/12.6.0 2>/dev/null || true
source /home/adp232/phd-research-gpu-env/bin/activate
SEED=${SEED:-42}
NAME=p2_ccsarch_v2_seed${SEED}.pt
echo "git=$(cat GIT_REV 2>/dev/null) seed=$SEED split_sha=$(sha256sum data/split_v2.json | cut -c1-12) start=$(date)"
nvidia-smi --query-gpu=name --format=csv,noheader
python3 -u -m src.training.train --config configs/dualhead_v2_large.yaml --seed $SEED \
    --amp --num-workers 4 --save-name $NAME
rc=$?
echo "EFFECT: rc=$rc ckpt=$(ls -la checkpoints/$NAME 2>&1) end=$(date)"
python3 - <<PY
import torch; c=torch.load('checkpoints/$NAME', map_location='cpu', weights_only=False)
print('EFFECT: epoch', c.get('epoch'), 'val_f1', c.get('val_f1'), 'split_sha256', str(c.get('split_sha256'))[:12], 'schema', c.get('split_schema'))
PY
