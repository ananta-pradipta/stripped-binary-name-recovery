#!/bin/bash
#SBATCH --job-name=cs785-eval --account=hz79 --partition=gpu --qos=standard
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:a100_10g:1 --mem=64G --time=36:00:00
#SBATCH --array=0-23
#SBATCH --output=/project/hz79/_shared/cs785/dh2/blens_ours_v2/pt_chunks/palmtree_%A_%a.out
# PalmTree encode of BLens ours-v2 in 24 load-balanced chunks (single-job pass projected ~72 h > 48 h wall, writes only at end).
WS=/project/hz79/_shared/cs785/dh2/blens_ours_v2
BUE=/project/hz79/_shared/cs785/baselines/blens_user_env
cd $BUE
module load CUDA/12.6.0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
C=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
OUT=$WS/pt_chunks/palmtree_chunk_$C.pkl
[ -s $OUT ] && { echo "EFFECT: $OUT already present, skip"; exit 0; }
./envs/BLens/bin/python palmtree_chunk.py $WS/pt_chunks/chunk_$C.txt $WS/xflBlensXProjectData $BUE/palmtree_weights $OUT.tmp 2>&1 | grep --line-buffered -avE "WARNING|angr\." 
rc=${PIPESTATUS[0]}; [ $rc = 0 ] && [ -s $OUT.tmp ] && mv $OUT.tmp $OUT
echo "EFFECT: chunk $C rc=$rc $(ls -la $OUT 2>&1 | awk '{print $5}')"; exit $rc
