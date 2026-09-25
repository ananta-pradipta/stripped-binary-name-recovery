#!/bin/bash
#SBATCH --job-name=cs785-eval --account=hz79 --partition=general --qos=standard
#SBATCH --cpus-per-task=4 --mem=48G --time=04:00:00
#SBATCH --output=/project/hz79/_shared/cs785/dh2/slurm/prov_build_%j.out
set -uo pipefail
cd /project/hz79/_shared/cs785/dh2
source /home/adp232/phd-research-gpu-env/bin/activate
python3 -u scripts/ctx_checks/provenance_build.py || { echo "FATAL: provenance build failed"; exit 1; }
for m in su du; do n=$(wc -l < results/a4_ctx_${m}_dm/test.jsonl); echo "EFFECT: rows $m $n"; [ "$n" -gt 50000 ] || { echo "FATAL: $m rows $n"; exit 1; }; done
echo "EFFECT: prov_build rc=0"
