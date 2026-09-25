#!/bin/bash
#SBATCH --job-name=cs785-eval --account=hz79 --partition=gpu --qos=standard
#SBATCH --gres=gpu:a100:1 --cpus-per-task=8 --mem=100G --time=36:00:00
#SBATCH --output=/project/hz79/_shared/cs785/punstrip/slurm/ctx_train_random_excl_%j.out
cd /project/hz79/_shared/cs785/dh2
module load CUDA/12.6.0 2>/dev/null || true
source /home/adp232/phd-research-gpu-env/bin/activate
export HF_HOME=/project/hz79/_shared/cs785/baselines/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TAG=punstrip_ctxrandom_excl_v1; D=/project/hz79/_shared/cs785/punstrip/data_ctx/random_excl
test -s $D/train.jsonl -a -s $D/val.jsonl || { echo "FATAL: $D rows missing"; exit 1; }
n=$(wc -l < $D/train.jsonl); [ "$n" -eq 378060 ] || { echo "FATAL: train rows $n"; exit 1; }
test -d checkpoints/a4_$TAG/best && { echo "FATAL: checkpoint exists"; exit 1; }
python3 -u scripts/a4_train_codet5p.py --model /project/hz79/_shared/cs785/baselines/hf_local/codet5p-220m \
  --tag $TAG --data-dir $D --max-src 1280 --bf16 --seed 42 --bs 8 --accum 4 2>&1 | grep --line-buffered -av "it/s"
rc=${PIPESTATUS[0]}; echo "EFFECT: train $TAG rc=$rc $(ls -d checkpoints/a4_$TAG/best 2>&1)"; exit $rc
