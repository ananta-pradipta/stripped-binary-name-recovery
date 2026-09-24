#!/bin/bash
#SBATCH --job-name=cs785-eval --account=ACCOUNT --partition=gpu --qos=standard
#SBATCH --gres=gpu:a100_40g:1 --cpus-per-task=8 --mem=64G --time=24:00:00
#SBATCH --output=$WORKSPACE/dh2/slurm/a4_train_s43_%j.out
cd $WORKSPACE/dh2
module load CUDA/12.6.0 2>/dev/null || true
source $HOME/phd-research-gpu-env/bin/activate
export HF_HOME=$WORKSPACE/baselines/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TAG=${TAG:-codet5p220m_v1_seed43}; MODEL=${MODEL:-$WORKSPACE/baselines/hf_local/codet5p-220m}
python3 -u scripts/a4_train_codet5p.py --tag $TAG --model $MODEL --bf16 --bs 8 --accum 4 --max-src 1024 --epochs 3 --eval-every 2000 --seed ${SEED:-43} 2>&1 | grep --line-buffered -av "it/s"
rc=${PIPESTATUS[0]}; echo "EFFECT: rc=$rc $(ls -d checkpoints/a4_$TAG/best 2>&1)"; exit $rc
