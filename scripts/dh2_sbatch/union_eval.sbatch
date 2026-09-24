#!/bin/bash
#SBATCH --job-name=cs785-eval --account=hz79 --partition=general --qos=standard
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=16G --time=00:40:00
#SBATCH --output=/project/hz79/_shared/cs785/dh2/slurm/union_eval_%j.out
cd /project/hz79/_shared/cs785/dh2
source /home/adp232/phd-research-gpu-env/bin/activate
python3 -u scripts/union_eval.py; echo "EFFECT: rc=$?"
