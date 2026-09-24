#!/bin/bash
#SBATCH --job-name=cs785-eval --account=ACCOUNT --partition=general --qos=standard
#SBATCH --cpus-per-task=32 --mem=96G --time=48:00:00
#SBATCH --output=$WORKSPACE/dh2/blens_ours_v2/ghidra_%j.out
WS=$WORKSPACE/dh2/blens_ours_v2
cd $WS
xargs -a bins.txt -P 14 -I{} bash $WORKSPACE/dh2/blens_ours_v2/ghidra_worker.sh {}
echo "EFFECT: clap_jsons=$(ls clap_jsons/*.clap.json 2>/dev/null | wc -l)/$(wc -l < bins.txt)"
