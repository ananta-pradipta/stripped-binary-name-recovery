#!/bin/bash
#SBATCH --job-name=cs785-eval --account=hz79 --partition=gpu --qos=standard
#SBATCH --gres=gpu:a100:1 --exclude=n0001,n0002,n0089,n0091,n0026 --cpus-per-task=8 --mem=120G --time=1-06:00:00
#SBATCH --output=/project/hz79/_shared/cs785/dh2/slurm/ctx_train_random_%j.out
cd /project/hz79/_shared/cs785/dh2
module load CUDA/12.6.0 2>/dev/null || true
source /home/adp232/phd-research-gpu-env/bin/activate
export HF_HOME=/project/hz79/_shared/cs785/baselines/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TAG=codet5p220m_ctxrandom_dm_v1; MODEL=/project/hz79/_shared/cs785/baselines/hf_local/codet5p-220m
D=results/a4_ctx_random_dm
test -s $D/train.jsonl -a -s $D/val.jsonl || { echo "FATAL: $D/*.jsonl missing"; exit 1; }
n=$(wc -l < $D/train.jsonl); [ "$n" -eq 190133 ] || { echo "FATAL: train rows $n != 190133"; exit 1; }
python3 -u scripts/a4_train_codet5p.py --tag $TAG --model $MODEL --data-dir $D --bf16 --bs 4 --accum 8 --max-src 1280 --epochs 3 --eval-every 2000 2>&1 | grep --line-buffered -av "it/s"
rc=${PIPESTATUS[0]}; echo "EFFECT: rc=$rc $(ls -d checkpoints/a4_$TAG/best 2>&1)"; exit $rc
