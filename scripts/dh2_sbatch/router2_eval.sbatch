#!/bin/bash
#SBATCH --job-name=cs785-eval --account=ACCOUNT --partition=general --qos=standard
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=24G --time=01:00:00
#SBATCH --output=$WORKSPACE/dh2/slurm/router2_%j.out
cd $WORKSPACE/dh2
source $HOME/phd-research-gpu-env/bin/activate
TAG=${TAG:-codet5p220m_v1}
python3 -u scripts/router2_eval.py --a4 results/a4_$TAG/val_test_symgen_holdout_preds.tsv --out results/router2_a1a_$TAG.json; echo "EFFECT: rc=$?"
