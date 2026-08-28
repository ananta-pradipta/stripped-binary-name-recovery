#!/bin/bash
#SBATCH --job-name=cs785-eval --account=hz79 --partition=general --qos=standard
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=64G --time=02:00:00
#SBATCH --output=/project/hz79/_shared/cs785/dh2/blens_ours_v2/pt_chunks/merge_%j.out
WS=/project/hz79/_shared/cs785/dh2/blens_ours_v2
BUE=/project/hz79/_shared/cs785/baselines/blens_user_env
cd $BUE
./envs/BLens/bin/python - <<PY
import pickle, glob, os
WS="$WS"; files=sorted(glob.glob(f"{WS}/pt_chunks/palmtree_chunk_*.pkl"))
assert len(files)==24, f"only {len(files)} chunk pickles present"
emb={}
for f in files:
    d=pickle.load(open(f,"rb")); n0=len(emb); emb.update(d); print(os.path.basename(f), len(d), "dup", n0+len(d)-len(emb), flush=True)
clap=pickle.load(open(f"{WS}/embedding/clap","rb"))
both=len(set(emb)&set(clap)); print("palmtree", len(emb), "clap", len(clap), "intersection", both, flush=True)
tmp=f"{WS}/embedding/palmtree.tmp"
pickle.dump(emb, open(tmp,"wb")); os.replace(tmp, f"{WS}/embedding/palmtree")
print("EFFECT: merged palmtree", len(emb), "entries; clap", len(clap), "intersection", both)
PY
rc=$?; echo "EFFECT: merge rc=$rc $(ls -la $WS/embedding/palmtree 2>&1 | awk '{print $5}')"; exit $rc
