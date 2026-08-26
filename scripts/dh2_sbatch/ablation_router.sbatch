#!/bin/bash
#SBATCH --job-name=cs785-eval --account=hz79 --partition=general --qos=standard
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=24G --time=02:00:00
#SBATCH --output=/project/hz79/_shared/cs785/dh2/slurm/ablation_router_%j.out
cd /project/hz79/_shared/cs785/dh2
source /home/adp232/phd-research-gpu-env/bin/activate
python3 -u scripts/ablation_router.py --a4 results/a4_codet5p220m_v1/val_test_symgen_holdout_preds.tsv --out results/ablation_router_a4v1.json; echo "EFFECT: rc=$?"
