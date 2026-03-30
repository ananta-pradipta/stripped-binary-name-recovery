#!/bin/bash -l
#SBATCH --partition=course_gpu
#SBATCH --account=2026-spring-cs-785-hz79-adp232
#SBATCH --qos=course
#SBATCH --time=01:00:00
#SBATCH --job-name=cs785-eval-test
#SBATCH --output=%x.%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100_10g:1

cd /course/2026/spring/cs/785/hz79/adp232/cs785

module load bright
module load python3
source ../cs785-env/bin/activate

echo "=== Evaluating val + test splits with full metrics ==="
python3 -u scripts/eval_test.py checkpoints/best_model.pt \
  --config configs/optimized_large.yaml \
  --amp \
  --splits val test

echo "=== Done. Exit code: $? ==="
