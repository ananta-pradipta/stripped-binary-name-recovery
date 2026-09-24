#!/bin/bash
#SBATCH --job-name=cs785-eval --account=ACCOUNT --partition=general --qos=standard
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=16G --time=00:40:00
#SBATCH --output=$WORKSPACE/dh2/slurm/union_eval_%j.out
cd $WORKSPACE/dh2
source $HOME/phd-research-gpu-env/bin/activate
python3 -u scripts/union_eval.py; echo "EFFECT: rc=$?"
