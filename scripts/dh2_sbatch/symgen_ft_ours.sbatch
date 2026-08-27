#!/bin/bash
#SBATCH --job-name=cs785-eval --account=hz79 --partition=gpu --qos=standard
#SBATCH --gres=gpu:a100:4 --cpus-per-task=16 --mem=128G --time=30:00:00
#SBATCH --output=/project/hz79/_shared/cs785/dh2/slurm/symgen_ft_ours_%j.out
# SymGen (CodeLlama-34B-Instruct + LoRA) fine-tuned on OUR frozen train tier (v2 protocol) — fairness item 1.
# Config = April run (job 909652): 4xA100 DDP, batch 128 = micro 8 x accum 4 x 4 GPUs, LoRA r=8 a=16 q_proj/v_proj, 1 epoch.
cd /project/hz79/_shared/cs785/baselines/SymGen/scripts/model_training
module load CUDA/12.6.0
source /project/hz79/_shared/cs785/baselines/symgen_ft_env/bin/activate
export HF_HOME=/project/hz79/_shared/cs785/baselines/hf_cache HF_HUB_OFFLINE=1 WANDB_DISABLED=true
DATA=/project/hz79/_shared/cs785/dh2/symgen_v2/full_input.json
OUT=/project/hz79/_shared/cs785/baselines/SymGen/lora_weights_ours_v2
test -s $DATA || { echo "FATAL: $DATA missing"; exit 1; }
torchrun --nproc_per_node=4 finetune.py --base_model codellama/CodeLlama-34b-Instruct-hf --data_path $DATA --output_dir $OUT \
  --batch_size 128 --micro_batch_size 8 --num_epochs 1 --learning_rate 3e-4 --cutoff_len 256 --val_set_size 0 \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 --lora_target_modules "[q_proj,v_proj]" --train_on_inputs False --group_by_length True
rc=$?; echo "EFFECT: rc=$rc $(ls $OUT 2>&1 | head -3)"; exit $rc
