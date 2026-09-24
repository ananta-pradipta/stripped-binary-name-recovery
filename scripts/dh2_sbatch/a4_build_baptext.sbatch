#!/bin/bash
#SBATCH --job-name=cs785-eval --account=hz79 --partition=general --qos=standard
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=64G --time=04:00:00
#SBATCH --output=/project/hz79/_shared/cs785/dh2/slurm/a4_build_baptext_%j.out
cd /project/hz79/_shared/cs785/dh2
source /home/adp232/phd-research-gpu-env/bin/activate
NPROC=16 python3 -u scripts/a4_build_baptext.py; rc=$?
head -c 1500 results/a4_baptext/val.jsonl; echo; wc -l results/a4_baptext/*.jsonl
python3 - <<PY
# token-length audit with the CodeT5+ tokenizer (offline), val tier
import json, os
os.environ['HF_HUB_OFFLINE']='1'
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained('/project/hz79/_shared/cs785/baselines/hf_local/codet5p-220m')
L=[len(tok(json.loads(l)['code']).input_ids) for i,l in enumerate(open('results/a4_baptext/val.jsonl')) if i<3000]
L.sort(); q=lambda p:L[int(p*(len(L)-1))]
print(f"EFFECT: val token lengths (n={len(L)}): p50 {q(.5)} p90 {q(.9)} max {L[-1]}; <=1024: {sum(1 for x in L if x<=1024)/len(L):.1%} <=1536: {sum(1 for x in L if x<=1536)/len(L):.1%} <=2048: {sum(1 for x in L if x<=2048)/len(L):.1%}")
PY
echo "EFFECT: rc=$rc"
