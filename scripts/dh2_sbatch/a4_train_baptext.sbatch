#!/bin/bash
#SBATCH --job-name=cs785-eval --account=ACCOUNT --partition=gpu --qos=standard
#SBATCH --gres=gpu:a100:1 --exclude=n0001,n0002,n0089,n0091,n0026 --cpus-per-task=8 --mem=120G --time=1-06:00:00
#SBATCH --output=$WORKSPACE/dh2/slurm/a4_train_baptext_%j.out
# Ghidra-vs-BAP representation control: identical CodeT5+ 220M recipe as A4 run 1, input = linearised BAP-IR text.
cd $WORKSPACE/dh2
module load CUDA/12.6.0 2>/dev/null || true
source $HOME/phd-research-gpu-env/bin/activate
export HF_HOME=$WORKSPACE/baselines/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TAG=${TAG:-codet5p220m_baptext_v1}; MODEL=$WORKSPACE/baselines/hf_local/codet5p-220m
MAXSRC=${MAXSRC:-1024}; BS=${BS:-8}; ACCUM=${ACCUM:-4}
test -s results/a4_baptext/train.jsonl -a -s results/a4_baptext/val.jsonl || { echo "FATAL: results/a4_baptext/*.jsonl missing"; exit 1; }
python3 -u scripts/a4_train_codet5p.py --tag $TAG --model $MODEL --data-dir results/a4_baptext --bf16 --bs $BS --accum $ACCUM --max-src $MAXSRC --epochs 3 --eval-every 2000 2>&1 | grep --line-buffered -av "it/s"
rc=${PIPESTATUS[0]}; echo "EFFECT: rc=$rc $(ls -d checkpoints/a4_$TAG/best 2>&1)"; exit $rc
