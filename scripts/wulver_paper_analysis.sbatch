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

cd <project-root>
source <project-root>-env/bin/activate

echo "=== Paper Analysis (Error Analysis + Statistical Significance) ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# --- Script 1: Error Analysis ---
echo "=========================================="
echo "RUNNING: Error Analysis"
echo "=========================================="
python3 scripts/error_analysis.py \
    checkpoints/best_model.pt \
    --config configs/optimized_large.yaml \
    --amp \
    --demo \
    --save results/error_analysis.json

echo ""
echo "=========================================="
echo "RUNNING: Statistical Significance"
echo "=========================================="
python3 scripts/statistical_significance.py \
    checkpoints/best_model.pt \
    --config configs/optimized_large.yaml \
    --amp \
    --demo \
    --n-bootstrap 1000 \
    --save results/statistical_significance.json

echo ""
echo "=== Done ==="
echo "Date: $(date)"
