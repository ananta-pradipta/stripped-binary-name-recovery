#!/bin/bash
#SBATCH --job-name=cs785-eval --account=ACCOUNT --partition=gpu --qos=standard
#SBATCH --gres=gpu:a100:1 --exclude=n0001,n0002,n0089,n0091,n0026 --cpus-per-task=8 --mem=128G --time=24:00:00
#SBATCH --output=$WORKSPACE/dh2/blens_ours_v2/train_resume_%j.out
# Follow-on for train.sbatch (1200059): COMBO 80 ep finished; LORD OOM'd at step 0 on a MIG a100_40g slice (n0002, shared
# card). LORD (137M, batch 512) needs a full 80 GB a100 node like the April run. Continue LORD from the latest checkpoint
# (trainLORD -loadEpoch N skips epochs <= N; ckpt every `interval`=4 epochs), then inferBest. Idempotent: resubmit with
# --dependency=afterany:<self> to cover >24 h; a real inference log (>=1000 target: lines) makes later legs no-ops.
set -uo pipefail
module load CUDA/12.6.0
DATA=$WORKSPACE/dh2/blens_ours_v2
XP=$DATA/xp/ours-v2
PY=$WORKSPACE/baselines/blens_user_env/envs/BLens/bin/python
cd $WORKSPACE/baselines/blens
valid_log() { for f in $XP/LORD-inference-logs-test-*.txt; do [ -f "$f" ] && [ "$(grep -c '^target:' "$f")" -ge 1000 ] && echo "$f" && return 0; done; return 1; }
if L=$(valid_log); then echo "EFFECT: valid inference log already present ($(grep -c '^target:' $L) targets): $L"; exit 0; fi
for f in $XP/LORD-inference-logs-test-*.txt; do [ -f "$f" ] && { echo "stub inference log (no targets) -> $f.stub"; mv "$f" "$f.stub_$SLURM_JOB_ID"; }; done
test -s $XP/COMBO-Function-0 || { echo "FATAL: COMBO base missing in $XP (pretrain did not finish)"; ls -la $XP; exit 1; }
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
N=$(ls $XP/LORD-Function-0-epoch-* 2>/dev/null | sed -E 's/.*epoch-([0-9]+)$/\1/' | sort -n | tail -1)
if [ -n "$N" ]; then echo "RESUME LORD from epoch $N"; LOAD="-loadEpoch $N"; else echo "no LORD checkpoint: training LORD from scratch"; LOAD=""; fi
$PY RunExp.py --cross-project -data-dir $DATA -config ablation-c+p.json -d ours-v2 -train $LOAD; echo "=== trainLORD rc=$?"
ls $XP/LORD-Function-0-epoch-* 2>/dev/null | tail -2
$PY RunExp.py --cross-project -data-dir $DATA -config ablation-c+p.json -d ours-v2 -inferBest; rc=$?; echo "=== inferBest rc=$rc"
if L=$(valid_log); then echo "EFFECT: rc=$rc valid inference log $L ($(grep -c '^target:' $L) targets)"; exit 0; else echo "EFFECT: rc=$rc NO valid inference log"; ls -la $XP | tail -5; exit 1; fi
