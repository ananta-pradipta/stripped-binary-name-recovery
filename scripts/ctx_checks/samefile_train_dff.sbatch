#!/bin/bash
#SBATCH --job-name=cs785-eval --account=hz79 --partition=gpu --qos=standard
#SBATCH --gres=gpu:a100:1 --cpus-per-task=8 --mem=100G --time=1-00:00:00
#SBATCH --output=/project/hz79/_shared/cs785/dh2/slurm/samefile_train_dff_%j.out
cd /project/hz79/_shared/cs785/dh2
module load CUDA/12.6.0 2>/dev/null || true
source /home/adp232/phd-research-gpu-env/bin/activate
export HF_HOME=/project/hz79/_shared/cs785/baselines/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TAG=codet5p220m_ctxdff_dm_v1; MODEL=/project/hz79/_shared/cs785/baselines/hf_local/codet5p-220m; D=results/a4_ctx_dff_dm
test -s $D/train.jsonl -a -s $D/val.jsonl || { echo "FATAL: $D rows missing"; exit 1; }
test -d checkpoints/a4_$TAG/best && { echo "FATAL: checkpoint exists"; exit 1; }
python3 -u scripts/a4_train_codet5p.py --tag $TAG --model $MODEL --data-dir $D --bf16 --bs 4 --accum 8 --max-src 1280 --epochs 3 --eval-every 2000 2>&1 | grep --line-buffered -av "it/s"
rc=${PIPESTATUS[0]}; echo "EFFECT: train $TAG rc=$rc $(ls -d checkpoints/a4_$TAG/best 2>&1)"; exit $rc
