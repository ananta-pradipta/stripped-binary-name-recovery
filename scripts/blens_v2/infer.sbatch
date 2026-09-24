#!/bin/bash
#SBATCH --job-name=cs785-eval --account=ACCOUNT --partition=gpu --qos=standard
#SBATCH --gres=gpu:a100:1 --cpus-per-task=4 --mem=64G --time=04:00:00
#SBATCH --output=$WORKSPACE/dh2/blens_v2/infer_%j.out
set -uo pipefail
module load CUDA/12.6.0
WS=$WORKSPACE/dh2/blens_v2
PY=$WORKSPACE/baselines/blens_user_env/envs/BLens/bin/python
cd $WORKSPACE/baselines/blens
MODE=lord EPOCH=59 $PY $WS/run_infer_v2ftc.py
echo "EFFECT: rc=$?"
