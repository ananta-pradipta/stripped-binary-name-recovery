#!/bin/bash
#SBATCH --job-name=cs785-eval --account=hz79 --partition=general --qos=standard
#SBATCH --cpus-per-task=4 --mem=64G --time=06:00:00
#SBATCH --output=/project/hz79/_shared/cs785/dh2/slurm/samefile_build_%j.out
set -uo pipefail
cd /project/hz79/_shared/cs785/dh2
source /home/adp232/phd-research-gpu-env/bin/activate
python3 -u scripts/ctx_checks/samefile_build.py --tiers train,val,test || { echo "FATAL: samefile build failed"; exit 1; }
for c in sfo dff addrm nonem; do for t in train val test; do n=$(wc -l < results/a4_ctx_${c}_dm/$t.jsonl); echo "EFFECT: rows $c $t $n"; done; done
n=$(wc -l < results/a4_ctx_sfo_dm/train.jsonl); [ "$n" -gt 30000 ] || { echo "FATAL: too few matched train rows $n"; exit 1; }
for c in dff addrm nonem; do m=$(wc -l < results/a4_ctx_${c}_dm/train.jsonl); [ "$m" -eq "$n" ] || { echo "FATAL: row mismatch $c $m vs $n"; exit 1; }; done
echo "EFFECT: samefile_build rc=0"
