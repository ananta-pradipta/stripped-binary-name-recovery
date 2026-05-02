#!/bin/bash -l
#SBATCH --job-name=bfnr-eval
#SBATCH --partition=course_gpu
#SBATCH --account=2026-spring-cs-785-<account>-<user>
#SBATCH --qos=course
#SBATCH --gres=gpu:a100_10g:1
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=bfnr-eval.%j.out

cd <project-root>
source <project-root>-env/bin/activate

# Full evaluation report with k-NN hybrid as default inference
python3 scripts/eval_full.py \
    --checkpoint checkpoints/best_model.pt \
    --config configs/optimized_large.yaml \
    --amp \
    --sections info test demo samples ablation

echo ""
echo "=== Done ==="
echo "Date: $(date)"
