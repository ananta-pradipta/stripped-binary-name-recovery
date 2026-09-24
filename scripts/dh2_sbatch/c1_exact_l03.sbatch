#!/bin/bash
#SBATCH --job-name=cs785-eval --account=ACCOUNT --partition=gpu --qos=standard
#SBATCH --gres=gpu:a100:1 --cpus-per-task=8 --mem=120G --time=24:00:00
#SBATCH --output=$WORKSPACE/dh2/slurm/c1_exact_l03_%j.out
cd $WORKSPACE/dh2
module load CUDA/12.6.0 2>/dev/null || true
source $HOME/phd-research-gpu-env/bin/activate
CFG=${CFG:-configs/dualhead_v2_strings.yaml}; KNN=${KNN:-results/emb_p3a}; NW=${NW:-4}
python3 -u -m src.training.train --config $CFG --seed 42 --amp --num-workers $NW --contrastive-mode exact --contrastive-weight 0.3 --hard-neg-knn $KNN --save-name c1_exact_l03_seed42.pt 2>&1 | grep --line-buffered -av "it/s"
rc=${PIPESTATUS[0]}; echo "EFFECT: rc=$rc $(ls -la checkpoints/c1_exact_l03_seed42.pt 2>&1)"; exit $rc
