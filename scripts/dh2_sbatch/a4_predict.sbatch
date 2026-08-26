#!/bin/bash
#SBATCH --job-name=cs785-eval --account=hz79 --partition=gpu --qos=standard
#SBATCH --gres=gpu:a100_40g:1 --cpus-per-task=8 --mem=64G --time=08:00:00
#SBATCH --output=/project/hz79/_shared/cs785/dh2/slurm/a4_predict_%j.out
cd /project/hz79/_shared/cs785/dh2
module load CUDA/12.6.0 2>/dev/null || true
source /home/adp232/phd-research-gpu-env/bin/activate
export HF_HOME=/project/hz79/_shared/cs785/baselines/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
TAG=${TAG:-codet5p220m_v1}
test -d checkpoints/a4_$TAG/best || { echo "FATAL: checkpoints/a4_$TAG/best missing"; exit 1; }
python3 -u scripts/a4_predict.py checkpoints/a4_$TAG/best --tag $TAG --tiers val test --rows results/a4_ft/symgen_holdout.jsonl --bf16 --bs 64 ${EXTRA} 2>&1 | grep --line-buffered -av "it/s"
rc=${PIPESTATUS[0]}; echo "EFFECT: rc=$rc $(ls results/a4_$TAG/val_test_symgen_holdout_eval.json 2>&1)"; exit $rc
