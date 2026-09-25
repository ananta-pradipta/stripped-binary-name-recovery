#!/bin/bash
#SBATCH --job-name=cs785-eval --account=hz79 --partition=general --qos=standard
#SBATCH --cpus-per-task=4 --mem=32G --time=03:00:00
#SBATCH --output=/project/hz79/_shared/cs785/punstrip/slurm/ctx_build_%j.out
set -uo pipefail
cd /project/hz79/_shared/cs785/dh2
source /home/adp232/phd-research-gpu-env/bin/activate
python3 -u scripts/ctx_checks/punstrip_build_ctxvar.py --modes win10,win5,win20,callgraph,random,empty --splits test || { echo "FATAL: punstrip ctx build failed"; exit 1; }
python3 -u scripts/ctx_checks/tu_locality.py --layout /project/hz79/_shared/cs785/punstrip/data_ctx/layout_test.json --root /project/hz79/_shared/cs785/punstrip/rebuild/dbg --out /project/hz79/_shared/cs785/punstrip/data_ctx/tu_locality_test.json || { echo "FATAL: locality failed"; exit 1; }
echo "EFFECT: punstrip ctx build rc=0"
