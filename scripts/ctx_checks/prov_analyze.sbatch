#!/bin/bash
#SBATCH --job-name=cs785-eval --account=hz79 --partition=general --qos=standard
#SBATCH --cpus-per-task=2 --mem=32G --time=02:00:00
#SBATCH --output=/project/hz79/_shared/cs785/dh2/slurm/prov_analyze_%j.out
cd /project/hz79/_shared/cs785/dh2
source /home/adp232/phd-research-gpu-env/bin/activate
python3 -u scripts/ctx_checks/provenance_analyze.py; rc=$?; echo "EFFECT: prov_analyze rc=$rc"; exit $rc
