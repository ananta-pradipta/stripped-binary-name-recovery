#!/bin/bash
#SBATCH --job-name=cs785-eval --account=hz79 --partition=general --qos=standard
#SBATCH --cpus-per-task=4 --mem=32G --time=03:00:00
#SBATCH --output=/project/hz79/_shared/cs785/punstrip/slurm/score_extra_final_%j.out
# Scores whichever extra systems have landed (SymGen-34B LoRA on Punstrip-train; BLens 2a retrain) next to ours and the
# published baselines on one common key set with both scorers. Safe to re-run; skips a system whose outputs are incomplete.
cd /project/hz79/_shared/cs785/punstrip
ARGS=""
# SymGen: all three test shards must be complete (predict.py appends incrementally, so check counts, not existence)
SG_OK=1
for k in 0 1 2; do
  N=$(python3 -c "import json;print(len(json.load(open(\"symgen/test_shards/shard_$k.json\"))))")
  P=$( [ -s symgen/results_test/shard_$k/predicted_function_name.json ] && python3 -c "import json;print(len(json.load(open(\"symgen/results_test/shard_$k/predicted_function_name.json\"))))" || echo 0 )
  echo "symgen shard $k: $P / $N"; [ "$P" = "$N" ] || SG_OK=0
done
[ $SG_OK = 1 ] && ARGS="$ARGS --symgen-dir symgen34b_lora_punstrip=$PWD/symgen" || echo "SKIP SymGen (incomplete)"
# BLens: pick the newest LORD test inference log, require 23,875 target lines
BL=$(ls -t blens/data/xp/punstrip/LORD-inference-logs-test-*.txt 2>/dev/null | head -1)
if [ -n "$BL" ] && [ "$(grep -c "^target:" "$BL")" = "23875" ]; then ARGS="$ARGS --blens-log blens_retrain_2a=$BL"; else echo "SKIP BLens (log missing/incomplete: ${BL:-none})"; fi
[ -n "$ARGS" ] || { echo "FATAL: nothing to score"; exit 1; }
echo "EFFECT: scoring with:$ARGS"
OUT=$PWD/results/system /project/hz79/_shared/cs785/tools/mm/root/envs/blensnlp/bin/python -u scripts/punstrip_score_extra.py $ARGS --self-test --tag final
rc=$?; echo "EFFECT: score_extra_final rc=$rc $(ls -la results/system/score_report_extra_final.json 2>&1)"; exit $rc
