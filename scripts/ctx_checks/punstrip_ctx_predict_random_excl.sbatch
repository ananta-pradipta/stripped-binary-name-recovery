#!/bin/bash
#SBATCH --job-name=cs785-eval --account=hz79 --partition=gpu --qos=standard
#SBATCH --gres=gpu:a100:1 --cpus-per-task=8 --mem=64G --time=06:00:00
#SBATCH --output=/project/hz79/_shared/cs785/punstrip/slurm/ctx_predict_random_excl_%j.out
cd /project/hz79/_shared/cs785/dh2
module load CUDA/12.6.0 2>/dev/null || true
source /home/adp232/phd-research-gpu-env/bin/activate
export HF_HOME=/project/hz79/_shared/cs785/baselines/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
TAG=punstrip_ctxrandom_excl_v1; D=/project/hz79/_shared/cs785/punstrip/data_ctx/random_excl
test -d checkpoints/a4_$TAG/best || { echo "FATAL: checkpoint missing"; exit 1; }
python3 -u scripts/a4_predict.py checkpoints/a4_$TAG/best --tag $TAG --tiers --rows $D/val.jsonl $D/test.jsonl --max-src 1280 --bf16 --bs 32 2>&1 | grep --line-buffered -av "it/s"
rc=${PIPESTATUS[0]}; echo "EFFECT: predict $TAG rc=$rc $(ls -la results/a4_$TAG/val_test_eval.json 2>&1)"; exit $rc
