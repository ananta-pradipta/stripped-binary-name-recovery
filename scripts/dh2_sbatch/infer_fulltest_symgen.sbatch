#!/bin/bash -l
#SBATCH --job-name=cs785-eval --account=hz79 --partition=gpu --qos=standard
#SBATCH --gres=gpu:a100:1 --exclude=n0001,n0002,n0089,n0091,n0026 --cpus-per-task=8 --mem=80G --time=10:00:00
#SBATCH --output=/project/hz79/_shared/cs785/dh2/symgen_v2/infer_fulltest_%A_%a.out
# Full-test-tier SymGen inference (CodeLlama-34B + LoRA retrained on our train tier), one shard (<=8,000 fns, ~2.4-3.4 s/fn) per array task.
set -euo pipefail
export HF_HOME=/project/hz79/_shared/cs785/baselines/hf_cache
export TRANSFORMERS_CACHE=/project/hz79/_shared/cs785/baselines/hf_cache
module load bright 2>/dev/null || true
module load CUDA/12.6.0
source /project/hz79/_shared/cs785/baselines/symgen_ft_env/bin/activate
SYMGEN=/project/hz79/_shared/cs785/baselines/SymGen
WS=/project/hz79/_shared/cs785/dh2/symgen_v2
K=$SLURM_ARRAY_TASK_ID; IN=$WS/fulltest_shards/shard_$K.json; OUT=$WS/results_fulltest/shard_$K
test -s $SYMGEN/lora_weights_ours_v2/adapter_model.bin || { echo "FATAL: adapter missing"; exit 1; }
test -s $IN || { echo "FATAL: shard $IN missing"; exit 1; }
N=$(python3 -c "import json;print(len(json.load(open('$IN'))))")
if [ -s $OUT/predicted_function_name.json ] && [ "$(python3 -c "import json;print(len(json.load(open('$OUT/predicted_function_name.json'))))")" = "$N" ]; then echo "EFFECT: shard $K already complete ($N preds)"; exit 0; fi
mkdir -p $OUT; echo "shard $K: $N inputs -> $OUT"
cd $SYMGEN/scripts/model_training
python3 predict.py --input_path="$IN" --output_dir="$OUT" --base_model="codellama/CodeLlama-34b-Instruct-hf" --lora_weights="$SYMGEN/lora_weights_ours_v2"
P=$(python3 -c "import json;print(len(json.load(open('$OUT/predicted_function_name.json'))))"); echo "EFFECT: shard $K preds=$P of $N"; [ "$P" = "$N" ]
