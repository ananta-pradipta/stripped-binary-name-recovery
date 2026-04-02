#!/bin/bash -l
#SBATCH --job-name=cs785-eval
#SBATCH --partition=course_gpu
#SBATCH --account=2026-spring-cs-785-hz79-adp232
#SBATCH --qos=course
#SBATCH --gres=gpu:a100_10g:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=cs785-knn.%j.out

echo "=== k-NN Hybrid Evaluation ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

cd /course/2026/spring/cs/785/hz79/adp232/cs785
source /course/2026/spring/cs/785/hz79/adp232/cs785-env/bin/activate

echo ""
echo "=== Step 1: Test set — threshold sweep ==="
python3 scripts/eval_knn_hybrid.py checkpoints/best_model.pt --sweep --amp --splits test

echo ""
echo "=== Step 2: Val set — threshold sweep ==="
python3 scripts/eval_knn_hybrid.py checkpoints/best_model.pt --sweep --amp --splits val

echo ""
echo "=== Done ==="
echo "Date: $(date)"
