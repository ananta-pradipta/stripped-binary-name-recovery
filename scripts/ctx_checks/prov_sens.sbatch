#!/bin/bash
#SBATCH --job-name=cs785-eval --account=hz79 --partition=gpu --qos=standard
#SBATCH --gres=gpu:a100:1 --cpus-per-task=8 --mem=64G --time=06:00:00
#SBATCH --array=0-1
#SBATCH --output=/project/hz79/_shared/cs785/dh2/slurm/prov_sens_%A_%a.out
cd /project/hz79/_shared/cs785/dh2
module load CUDA/12.6.0 2>/dev/null || true
source /home/adp232/phd-research-gpu-env/bin/activate
export HF_HOME=/project/hz79/_shared/cs785/baselines/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
MODES=(su du); M=${MODES[$SLURM_ARRAY_TASK_ID]}
CK=checkpoints/a4_codet5p220m_modctx_dm_v1/best
test -s results/a4_ctx_${M}_dm/test.jsonl || { echo "FATAL: rows for $M missing"; exit 1; }
python3 -u scripts/a4_predict.py $CK --tag modctx_dm_sens_$M --tiers --rows results/a4_ctx_${M}_dm/test.jsonl --max-src 1280 --bf16 --bs 32 2>&1 | grep --line-buffered -av "it/s"
rc=${PIPESTATUS[0]}
echo "EFFECT: prov sens $M rc=$rc $(python3 -c "import json;d=json.load(open('results/a4_modctx_dm_sens_$M/test_eval.json'))['tiers']['test'];print('F1',round(d['scored']['f1'],4),'macro',round(d['macro_pkg']['f1'],4),'n',d['scored']['n'])" 2>&1)"
exit $rc
