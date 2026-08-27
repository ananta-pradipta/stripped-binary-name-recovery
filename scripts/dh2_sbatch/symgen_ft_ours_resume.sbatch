#!/bin/bash
#SBATCH --job-name=cs785-eval --account=hz79 --partition=gpu --qos=standard
#SBATCH --gres=gpu:a100:4 --cpus-per-task=16 --mem=300G --time=30:00:00
#SBATCH --output=/project/hz79/_shared/cs785/dh2/slurm/symgen_ft_ours_resume_%j.out
# Resume leg for symgen_ft_ours.sbatch: 40 GB cards force micro-batch 2 (~120 s/step x 1485 steps ~ 49.5 h) > 30 h wall.
# Picks the latest COMPLETE Trainer checkpoint (trainer_state.json + optimizer.pt + adapter weights) and resumes
# optimizer/scheduler/step via the patched finetune.py (keeps the resume dir when trainer_state.json exists).
cd /project/hz79/_shared/cs785/baselines/SymGen/scripts/model_training
module load CUDA/12.6.0
source /project/hz79/_shared/cs785/baselines/symgen_ft_env/bin/activate
export HF_HOME=/project/hz79/_shared/cs785/baselines/hf_cache TRANSFORMERS_CACHE=/project/hz79/_shared/cs785/baselines/hf_cache HF_HUB_OFFLINE=1 WANDB_DISABLED=true TORCHINDUCTOR_COMPILE_THREADS=1 OMP_NUM_THREADS=4
DATA=/project/hz79/_shared/cs785/dh2/symgen_v2/full_input.json
OUT=/project/hz79/_shared/cs785/baselines/SymGen/lora_weights_ours_v2
if [ -s $OUT/adapter_model.bin ] || [ -s $OUT/adapter_model.safetensors ]; then echo "EFFECT: final adapter already present in $OUT, nothing to resume"; exit 0; fi
CKPT=""
for d in $(ls -d $OUT/checkpoint-* 2>/dev/null | sort -t- -k2,2n -r); do
  if [ -s $d/trainer_state.json ] && [ -s $d/optimizer.pt ] && { [ -s $d/adapter_model.bin ] || [ -s $d/adapter_model.safetensors ]; }; then CKPT=$d; break; else echo "skip incomplete $d"; fi
done
test -n "$CKPT" || { echo "FATAL: no complete checkpoint under $OUT"; ls -la $OUT; exit 1; }
if [ ! -s $CKPT/adapter_model.bin ] && [ -s $CKPT/adapter_model.safetensors ]; then
  python -c 'from safetensors.torch import load_file;import torch,sys;torch.save(load_file(sys.argv[1]+"/adapter_model.safetensors"),sys.argv[1]+"/adapter_model.bin")' $CKPT
fi
echo "RESUME from $CKPT step=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]+"/trainer_state.json"))["global_step"])' $CKPT)"
torchrun --nproc_per_node=4 finetune.py --base_model codellama/CodeLlama-34b-Instruct-hf --data_path $DATA --output_dir $OUT \
  --batch_size 128 --micro_batch_size 2 --num_epochs 1 --learning_rate 3e-4 --cutoff_len 256 --val_set_size 0 \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 --lora_target_modules "[q_proj,v_proj]" --resume_from_checkpoint $CKPT
rc=$?; echo "EFFECT: rc=$rc $(ls $OUT 2>&1 | head -5)"; exit $rc
