#!/bin/bash -l
#SBATCH --partition=course_gpu
#SBATCH --account=<hpc-account>
#SBATCH --qos=course
#SBATCH --time=02:00:00
#SBATCH --job-name=bfnr-demo
#SBATCH --output=%x.%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100_10g:1

cd <project-root>

module load bright
module load python3
source ../bfnr-env/bin/activate

echo "=== Demo Eval (HPC — no BAP) ==="
python3 -u scripts/eval_demo_wulver.py checkpoints/best_model.pt

echo "=== Done. Exit code: $? ==="
