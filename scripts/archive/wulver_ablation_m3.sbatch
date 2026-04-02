#!/bin/bash -l
#SBATCH --partition=course_gpu
#SBATCH --account=2026-spring-cs-785-hz79-adp232
#SBATCH --qos=course
#SBATCH --time=04:00:00
#SBATCH --job-name=cs785-abl3
#SBATCH --output=%x.%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100_10g:1

cd /course/2026/spring/cs/785/hz79/adp232/cs785
module load bright
module load python3
source ../cs785-env/bin/activate

echo "=== Model 3: + External Call Encoder ==="
mkdir -p checkpoints/ablation_model3
python3 -u -m src.training.train --config configs/ablation_model3.yaml --seed 42 --amp --num-workers 4
echo "Done. Exit: $?"
