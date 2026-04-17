#!/bin/bash -l
#SBATCH --partition=course_gpu
#SBATCH --account=<hpc-account>
#SBATCH --qos=course
#SBATCH --time=04:00:00
#SBATCH --job-name=bfnr-abl2
#SBATCH --output=%x.%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100_10g:1

cd <project-root>
module load bright
module load python3
source ../bfnr-env/bin/activate

echo "=== Model 2: GAT + Decoder (no ext, no context) ==="
mkdir -p checkpoints/ablation_model2
python3 -u -m src.training.train --config configs/ablation_model2.yaml --seed 42 --amp --num-workers 4
echo "Done. Exit: $?"
