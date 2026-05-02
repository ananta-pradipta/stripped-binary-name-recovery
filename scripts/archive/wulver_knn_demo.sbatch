#!/bin/bash -l
#SBATCH --job-name=bfnr-eval
#SBATCH --partition=course_gpu
#SBATCH --account=2026-spring-cs-785-<account>-<user>
#SBATCH --qos=course
#SBATCH --gres=gpu:a100_10g:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=bfnr-eval.%j.out

echo "=== k-NN Reverse Hybrid Evaluation ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

cd <project-root>
source <project-root>-env/bin/activate

echo ""
echo "=== Reverse hybrid: test set ==="
python3 scripts/eval_knn_hybrid.py checkpoints/best_model.pt --sweep --amp --reverse --splits test

echo ""
echo "=== Reverse hybrid: demo set ==="
python3 scripts/eval_knn_hybrid.py checkpoints/best_model.pt --sweep --amp --reverse --demo

echo ""
echo "=== Done ==="
echo "Date: $(date)"
