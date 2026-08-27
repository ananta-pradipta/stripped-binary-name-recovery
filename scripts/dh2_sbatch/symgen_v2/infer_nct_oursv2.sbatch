#!/bin/bash -l
#SBATCH --job-name=cs785-eval --account=hz79 --partition=gpu --qos=standard
#SBATCH --gres=gpu:a100:1 --cpus-per-task=8 --mem=80G --time=08:00:00
#SBATCH --output=/project/hz79/_shared/cs785/dh2/symgen_v2/infer_c_%j.out
set -euo pipefail
export HF_HOME=/project/hz79/_shared/cs785/baselines/hf_cache
export TRANSFORMERS_CACHE=/project/hz79/_shared/cs785/baselines/hf_cache
module load bright 2>/dev/null || true
module load CUDA/12.6.0
source /project/hz79/_shared/cs785/baselines/symgen_ft_env/bin/activate
SYMGEN=/course/2026/spring/cs/785/hz79/adp232/cs785/baselines/SymGen
WS=/project/hz79/_shared/cs785/dh2/symgen_v2
python3 - <<PY
import json
inp=json.load(open("$WS/nct_seen_input.json")); meta=json.load(open("$WS/nct_seen_metadata.json"))
assert len(inp)==len(meta) and len(inp)>4000, (len(inp),len(meta))
assert all("[MASK]" in t["input"] for t in inp[:100])
print("preflight ok:",len(inp))
PY
cd $SYMGEN/scripts/model_training
python3 predict.py --input_path="$WS/nct_seen_input.json" --output_dir="$WS/results_nct_oursv2"   --base_model="codellama/CodeLlama-34b-Instruct-hf" --lora_weights="$SYMGEN/lora_weights_ours_v2"
echo "EFFECT: rc=$? preds=$(python3 -c "import json;print(len(json.load(open(\"$WS/results_nct/predicted_function_name.json\"))))" 2>&1)"
