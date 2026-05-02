#!/bin/bash -l
#SBATCH --job-name=bfnr-eval
#SBATCH --partition=course_gpu
#SBATCH --account=2026-spring-cs-785-<account>-<user>
#SBATCH --qos=course
#SBATCH --gres=gpu:a100_10g:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=bfnr-eval.%j.out

echo "=== k-NN Hybrid — Single Threshold (all metrics) ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

cd <project-root>
source <project-root>-env/bin/activate

echo ""
echo "=== Forward hybrid: demo, threshold=-0.02 ==="
python3 scripts/eval_knn_hybrid.py checkpoints/best_model.pt --amp --demo --threshold -0.02

echo ""
echo "=== Done ==="
echo "Date: $(date)"
