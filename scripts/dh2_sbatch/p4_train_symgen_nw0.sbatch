#!/bin/bash
#SBATCH --job-name=cs785-eval --account=hz79 --partition=gpu --qos=standard
#SBATCH --gres=gpu:a100:1 --cpus-per-task=8 --mem=200G --time=24:00:00
#SBATCH --output=/project/hz79/_shared/cs785/dh2/slurm/p4_sg_nw0_%j.out
cd /project/hz79/_shared/cs785/dh2
module load CUDA/12.6.0 2>/dev/null || true
source /home/adp232/phd-research-gpu-env/bin/activate
python3 -u -m src.training.train --config configs/dualhead_v2_strings_symgen.yaml --seed 42 --amp --num-workers 0 --save-name p4_strings_symgen_v3_seed42.pt
rc=$?; echo "EFFECT: rc=$rc $(ls -la checkpoints/p4_strings_symgen_v3_seed42.pt 2>&1)"; exit $rc
