#!/bin/bash
#SBATCH --job-name=cs785-eval --account=hz79 --partition=general --qos=standard
#SBATCH --cpus-per-task=4 --mem=48G --time=06:00:00
#SBATCH --output=/project/hz79/_shared/cs785/dh2/slurm/ctx_build2_%j.out
set -uo pipefail
cd /project/hz79/_shared/cs785/dh2
source /home/adp232/phd-research-gpu-env/bin/activate
python3 -u scripts/ctx_checks/a4_build_ctxvar.py --modes addrcall --tiers train,val,test || { echo "FATAL: addrcall build failed"; exit 1; }
python3 -u scripts/ctx_checks/a4_build_ctxvar.py --modes random_excl --tiers val,test --seed 1 --suffix _s1 || { echo "FATAL: random_excl s1 failed"; exit 1; }
python3 -u scripts/ctx_checks/a4_build_ctxvar.py --modes random_excl --tiers val,test --seed 2 --suffix _s2 || { echo "FATAL: random_excl s2 failed"; exit 1; }
python3 -u scripts/ctx_checks/punstrip_build_ctxvar.py --modes addrcall,random_excl --splits test --seed 1 --suffix _s1 || { echo "FATAL: punstrip s1 failed"; exit 1; }
python3 -u scripts/ctx_checks/punstrip_build_ctxvar.py --modes random_excl --splits test --seed 2 --suffix _s2 || { echo "FATAL: punstrip s2 failed"; exit 1; }
n=$(wc -l < results/a4_ctx_addrcall_dm/train.jsonl); [ "$n" -eq 190133 ] || { echo "FATAL: addrcall train rows $n"; exit 1; }
echo "EFFECT: ctx_build2 rc=0"
