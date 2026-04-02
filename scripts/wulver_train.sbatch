#!/bin/bash -l
#SBATCH --partition=course_gpu
#SBATCH --account=2026-spring-cs-785-hz79-adp232
#SBATCH --qos=course
#SBATCH --time=08:00:00
#SBATCH --job-name=cs785-train
#SBATCH --output=%x.%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1

# Navigate to project directory
cd /course/2026/spring/cs/785/hz79/adp232/cs785

# Load modules and activate Python environment
module load bright
module load python3
source ../cs785-env/bin/activate

echo "=========================================="
echo "  CS785 Training Job"
echo "=========================================="
echo "Date:       $(date)"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "GPU:        $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python:     $(python3 --version 2>&1)"
echo "PyTorch:    $(python3 -c 'import torch; print(torch.__version__)' 2>&1)"
echo "CUDA:       $(python3 -c 'import torch; print(torch.cuda.is_available())' 2>&1)"
echo "Time limit: $SLURM_TIMELIMIT"
echo "CPUs:       $SLURM_CPUS_PER_TASK"
echo "Memory:     $SLURM_MEM_PER_NODE MB"
echo "=========================================="
echo ""

# Train with pretrained encoder
# batch=32: proven best convergence (Val F1 0.7325 at epoch 46)
# AMP: mixed precision for memory efficiency
# num-workers 4: parallel data loading
python3 -u -m src.training.train \
  --config configs/optimized_large.yaml \
  --seed 42 \
  --pretrained-encoder checkpoints/pretrained_encoder.pt \
  --amp \
  --num-workers 4

echo ""
echo "=========================================="
echo "Training complete."
echo "Exit code: $?"
echo "End time:  $(date)"
echo "=========================================="
