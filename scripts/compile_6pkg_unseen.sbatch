#!/bin/bash
#SBATCH --job-name=bfnr-eval
#SBATCH --output=$WORKSPACE/slurm_logs/bfnr-compile-6pkg-unseen.%j.out
#SBATCH --account=ACCOUNT
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
module load bright 2>/dev/null || true
module load GCC/12.3.0 2>/dev/null || true

cd $WORKSPACE
bash scripts/compile_6pkg_unseen.sh

echo '=== Compile done ==='
date
ls -la data/cross_project/debug_unseen/ | head -30
ls -la data/cross_project/stripped_unseen/ | head -30
