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
#SBATCH --output=cs785-eval.%j.out

cd /course/2026/spring/cs/785/hz79/adp232/cs785
source /course/2026/spring/cs/785/hz79/adp232/cs785-env/bin/activate

echo "=== k-NN Ablation Study ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# Run k-NN ablation on test + demo sets
# k=1,3,5,10 x {decoder-only, knn-only, forward hybrid, reverse hybrid}
python3 scripts/eval_knn_ablation.py \
    checkpoints/best_model.pt \
    --config configs/optimized_large.yaml \
    --amp \
    --demo \
    --k-values 1 3 5 10 \
    --save results/knn_ablation.json

echo ""
echo "=== Done ==="
echo "Date: $(date)"
