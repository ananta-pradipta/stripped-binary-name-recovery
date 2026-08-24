#!/bin/bash
#SBATCH --job-name=cs785-eval --account=hz79 --partition=general --qos=standard
#SBATCH --cpus-per-task=32 --mem=96G --time=08:00:00
#SBATCH --output=/project/hz79/_shared/cs785/dh2/blens_v2/ghidra_%j.out
WS=/project/hz79/_shared/cs785/dh2/blens_v2
cd $WS
xargs -a bins.txt -P 14 -I{} bash /project/hz79/_shared/cs785/dh2/blens_v2/ghidra_worker.sh {}
echo "EFFECT: clap_jsons=$(ls clap_jsons/*.clap.json 2>/dev/null | wc -l)/$(wc -l < bins.txt)"
