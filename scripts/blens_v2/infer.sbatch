#!/bin/bash
#SBATCH --job-name=cs785-eval --account=hz79 --partition=gpu --qos=standard
#SBATCH --gres=gpu:a100:1 --cpus-per-task=4 --mem=64G --time=04:00:00
#SBATCH --output=/project/hz79/_shared/cs785/dh2/blens_v2/infer_%j.out
set -uo pipefail
module load CUDA/12.6.0
WS=/project/hz79/_shared/cs785/dh2/blens_v2
PY=/project/hz79/_shared/cs785/baselines/blens_user_env/envs/BLens/bin/python
cd /project/hz79/_shared/cs785/baselines/blens
MODE=lord EPOCH=59 $PY $WS/run_infer_v2ftc.py
echo "EFFECT: rc=$?"
