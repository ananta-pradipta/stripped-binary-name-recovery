#!/bin/bash
#SBATCH --job-name=cs785-eval --account=hz79 --partition=general --qos=standard
#SBATCH --cpus-per-task=4 --mem=64G --time=10:00:00
#SBATCH --output=/project/hz79/_shared/cs785/punstrip/slurm/ctx_build_train_%j.out
set -uo pipefail
cd /project/hz79/_shared/cs785/dh2
source /home/adp232/phd-research-gpu-env/bin/activate
python3 -u scripts/ctx_checks/punstrip_build_ctxvar.py --modes win10,none,random_excl,callgraph,addrcall --splits train,val,test --seed 1 || { echo "FATAL: punstrip trained-variant build failed"; exit 1; }
for m in none random_excl callgraph addrcall; do
  n=$(wc -l < /project/hz79/_shared/cs785/punstrip/data_ctx/$m/train.jsonl); [ "$n" -eq 378060 ] || { echo "FATAL: $m train rows $n != 378060"; exit 1; }
  t=$(wc -l < /project/hz79/_shared/cs785/punstrip/data_ctx/$m/test.jsonl); [ "$t" -eq 23873 ] || { echo "FATAL: $m test rows $t != 23873"; exit 1; }
done
echo "EFFECT: punstrip trained-variant build rc=0"
