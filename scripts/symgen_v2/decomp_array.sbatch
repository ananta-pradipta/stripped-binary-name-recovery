#!/bin/bash
#SBATCH --job-name=cs785-eval --account=ACCOUNT --partition=general --qos=standard
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=96G --time=12:00:00
#SBATCH --output=$WORKSPACE/dh2/symgen_v2/decomp_array_%j.out
WS=$WORKSPACE/dh2/symgen_v2
cd $WS
xargs -a binaries.txt -P 14 -I{} bash decomp_worker.sh {}
done=$(ls decomp/*.json 2>/dev/null | wc -l); tot=$(wc -l < binaries.txt)
echo "EFFECT: decomp_done=$done/$tot"
