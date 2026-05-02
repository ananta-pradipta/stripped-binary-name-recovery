#!/bin/bash -l
#SBATCH --partition=course_gpu
#SBATCH --account=2026-spring-cs-785-<account>-<user>
#SBATCH --qos=course
#SBATCH --time=02:00:00
#SBATCH --job-name=bfnr-eval
#SBATCH --output=%x.%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100_10g:1

# Navigate to project directory
cd <project-root>

# Load modules and activate Python environment
module load bright
module load python3
source ../bfnr-env/bin/activate

echo "=== Running Demo Eval ==="
python3 scripts/eval_demo_expanded.py checkpoints/best_model.pt

echo "=== Eval complete. Exit code: $? ==="
