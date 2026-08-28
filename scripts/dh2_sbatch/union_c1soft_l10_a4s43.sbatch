#!/bin/bash
#SBATCH --job-name=cs785-eval --account=hz79 --partition=general --qos=standard
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=24G --time=01:30:00
#SBATCH --output=/project/hz79/_shared/cs785/dh2/slurm/union_c1soft_l10_a4s43_%j.out
cd /project/hz79/_shared/cs785/dh2
source /home/adp232/phd-research-gpu-env/bin/activate
python3 -u scripts/union_eval.py --router results/router_c1soft_l10_interim_features.tsv --regime-dump results/c1soft_l10_interim_preds_greedy.tsv --a4 results/a4_codet5p220m_v1_seed43/val_test_symgen_holdout_preds.tsv --out results/union_c1soft_l10_interim_a4s43.json; echo "EFFECT: union rc=$?"
python3 -u scripts/router2_eval.py --router results/router_c1soft_l10_interim_features.tsv --regime-dump results/c1soft_l10_interim_preds_greedy.tsv --a4 results/a4_codet5p220m_v1_seed43/val_test_symgen_holdout_preds.tsv --out results/router2_c1soft_l10_interim_a4s43.json; echo "EFFECT: router2 rc=$?"
