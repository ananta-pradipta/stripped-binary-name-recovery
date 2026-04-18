#!/bin/bash
#SBATCH --job-name=bfnr-eval
#SBATCH --output=/mmfs1/project/hz79/_shared/cs785/slurm_logs/bfnr-compile-6pkg-unseen.%j.out
#SBATCH --account=hz79
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
set -e
module load bright 2>/dev/null
module load GCC/12.3.0 2>/dev/null

cd /mmfs1/project/hz79/_shared/cs785
bash scripts/compile_6pkg_unseen.sh

echo '=== Compile done ==='
date
ls -la data/cross_project/debug_unseen/ | head -30
ls -la data/cross_project/stripped_unseen/ | head -30
