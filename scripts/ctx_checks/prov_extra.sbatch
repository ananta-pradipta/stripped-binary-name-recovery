#!/bin/bash
#SBATCH --job-name=cs785-eval --account=hz79 --partition=general --qos=standard
#SBATCH --cpus-per-task=2 --mem=48G --time=04:00:00
#SBATCH --output=/project/hz79/_shared/cs785/dh2/slurm/prov_extra_%j.out
cd /project/hz79/_shared/cs785/dh2
source /home/adp232/phd-research-gpu-env/bin/activate
python3 -u scripts/ctx_checks/provenance_extra.py || { echo "FATAL: provenance_extra failed"; exit 1; }
python3 -u scripts/ctx_checks/provenance_analyze.py | grep -a 'T16\|T17\|EFFECT'; echo "EFFECT: prov_extra rc=0"
