#!/bin/bash
#SBATCH --job-name=cs785-eval --account=hz79 --partition=gpu --qos=standard
#SBATCH --gres=gpu:a100:1 --cpus-per-task=8 --mem=64G --time=08:00:00
#SBATCH --output=/project/hz79/_shared/cs785/dh2/blens_v2/encode_%j.out
set -uo pipefail
module load CUDA/12.6.0
WS=/project/hz79/_shared/cs785/dh2/blens_v2
BUE=/project/hz79/_shared/cs785/baselines/blens_user_env
cd $BUE
export HF_HOME=/project/hz79/_shared/cs785/baselines/hf_cache
./envs/BLens/bin/python $WS/extract_clap_v2ftc.py
echo "clap rc=$?"
./envs/BLens/bin/python palmtree_chunk.py $WS/bins.txt $WS/nlpData_v2ftc.pkl $BUE/palmtree_weights $WS/palmtree_v2ftc.pkl
echo "EFFECT: palmtree rc=$? $(ls -la $WS/palmtree_v2ftc.pkl 2>&1)"
