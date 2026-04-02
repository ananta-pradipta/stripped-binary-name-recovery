#!/bin/bash -l
#SBATCH --partition=course_gpu
#SBATCH --account=2026-spring-cs-785-hz79-adp232
#SBATCH --qos=course
#SBATCH --time=12:00:00
#SBATCH --job-name=cs785-ablation
#SBATCH --output=%x.%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_10g:1

cd /course/2026/spring/cs/785/hz79/adp232/cs785

module load bright
module load python3
source ../cs785-env/bin/activate

echo "=========================================="
echo "  Ablation Study — 3 Models Sequential"
echo "  Date: $(date)"
echo "  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "=========================================="

# Create checkpoint dirs
mkdir -p checkpoints/ablation_model2 checkpoints/ablation_model3 checkpoints/ablation_model4

# Model 2: GAT + Decoder (no ext calls, no context)
echo ""
echo "=========================================="
echo "  Model 2: GAT + Decoder (basic)"
echo "  Start: $(date)"
echo "=========================================="
python3 -u -m src.training.train \
  --config configs/ablation_model2.yaml \
  --seed 42 \
  --amp \
  --num-workers 4

echo "Model 2 done. Exit: $?. Time: $(date)"

# Model 3: + Ext Call Encoder
echo ""
echo "=========================================="
echo "  Model 3: + External Call Encoder"
echo "  Start: $(date)"
echo "=========================================="
python3 -u -m src.training.train \
  --config configs/ablation_model3.yaml \
  --seed 42 \
  --amp \
  --num-workers 4

echo "Model 3 done. Exit: $?. Time: $(date)"

# Model 4: + Callee/Caller Context
echo ""
echo "=========================================="
echo "  Model 4: + Callee/Caller Context"
echo "  Start: $(date)"
echo "=========================================="
python3 -u -m src.training.train \
  --config configs/ablation_model4.yaml \
  --seed 42 \
  --amp \
  --num-workers 4

echo "Model 4 done. Exit: $?. Time: $(date)"

echo ""
echo "=========================================="
echo "  All ablation models complete!"
echo "  End: $(date)"
echo "=========================================="
