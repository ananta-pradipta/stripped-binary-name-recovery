#!/bin/bash -l
#SBATCH --partition=course_gpu
#SBATCH --account=2026-spring-cs-785-<account>-<user>
#SBATCH --qos=course
#SBATCH --time=01:00:00
#SBATCH --job-name=bfnr-eval-test
#SBATCH --output=%x.%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100_10g:1

cd <project-root>

module load bright
module load python3
source ../bfnr-env/bin/activate

echo "=== Evaluating val + test splits with full metrics ==="
python3 -u scripts/eval_test.py checkpoints/best_model.pt \
  --config configs/optimized_large.yaml \
  --amp \
  --splits val test

echo "=== Done. Exit code: $? ==="
