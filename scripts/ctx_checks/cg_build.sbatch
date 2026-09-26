#!/bin/bash
#SBATCH --job-name=cs785-eval --account=hz79 --partition=general --qos=standard
#SBATCH --cpus-per-task=4 --mem=64G --time=08:00:00
#SBATCH --output=/project/hz79/_shared/cs785/dh2/slurm/cg_build_%j.out
set -uo pipefail
cd /project/hz79/_shared/cs785/dh2
source /home/adp232/phd-research-gpu-env/bin/activate
python3 -u scripts/ctx_checks/b0_strata.py || echo "WARN: b0_strata failed"
python3 -u scripts/ctx_checks/cg_variants_build.py --modes b1_role,b2_bal,b3_sig,b4_typed --tiers train,val,test || { echo "FATAL: cg variants build failed"; exit 1; }
for m in b1_role b2_bal b3_sig b4_typed; do n=$(wc -l < results/a4_ctx_${m}_dm/train.jsonl); [ "$n" -eq 190133 ] || { echo "FATAL: $m train rows $n"; exit 1; }; t=$(wc -l < results/a4_ctx_${m}_dm/test.jsonl); [ "$t" -eq 268136 ] || { echo "FATAL: $m test rows $t"; exit 1; }; done
echo "EFFECT: cg_build rc=0"
