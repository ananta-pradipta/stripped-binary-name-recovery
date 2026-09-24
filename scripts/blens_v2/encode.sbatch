#!/bin/bash
#SBATCH --job-name=cs785-eval --account=ACCOUNT --partition=gpu --qos=standard
#SBATCH --gres=gpu:a100:1 --cpus-per-task=8 --mem=64G --time=08:00:00
#SBATCH --output=$WORKSPACE/dh2/blens_v2/encode_%j.out
set -uo pipefail
module load CUDA/12.6.0
WS=$WORKSPACE/dh2/blens_v2
BUE=$WORKSPACE/baselines/blens_user_env
cd $BUE
export HF_HOME=$WORKSPACE/baselines/hf_cache
./envs/BLens/bin/python $WS/extract_clap_v2ftc.py
echo "clap rc=$?"
./envs/BLens/bin/python palmtree_chunk.py $WS/bins.txt $WS/nlpData_v2ftc.pkl $BUE/palmtree_weights $WS/palmtree_v2ftc.pkl
echo "EFFECT: palmtree rc=$? $(ls -la $WS/palmtree_v2ftc.pkl 2>&1)"
