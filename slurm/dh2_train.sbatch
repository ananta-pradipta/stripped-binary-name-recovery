#!/bin/bash -l
#SBATCH --job-name=cs785-eval
#SBATCH --account=ACCOUNT
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:00:00
#SBATCH --output=$WORKSPACE/dh2/slurm/%x.%j.out
# Usage: sbatch --export=CFG=configs/dualhead_v2_large.yaml,SEED=42,NAME=dh2_large_s42 slurm/dh2_train.sbatch
set -euo pipefail
cd $WORKSPACE/dh2
source $HOME/phd-research-gpu-env/bin/activate 2>/dev/null || source /course/2026/spring/cs/785/ACCOUNT/USER/cs785-env/bin/activate
export PYTHONPATH=$PWD:${PYTHONPATH:-}
CFG=${CFG:-configs/dualhead_v2_large.yaml}; SEED=${SEED:-42}; NAME=${NAME:-dh2_large_s${SEED}}
echo "=== dh2 train $NAME cfg=$CFG seed=$SEED $(date) host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
python3 - <<'PY'
import json, hashlib, sys, os
sys.path.insert(0, os.getcwd())
from scripts.preflight import Preflight
import yaml
cfg = yaml.safe_load(open(os.environ.get('CFG', 'configs/dualhead_v2_large.yaml')))
pf = Preflight('dh2 train')
sp = json.load(open(cfg['data']['split_file']))
tr = set(sp['train']); xp = set(sp.get('xproject', [])); vx = set(sp.get('val_xproj', []))
pf.count('train binaries', len(tr), minimum=300)
pf.count('xproject binaries', len(xp), minimum=20)
pf.count('val_xproj binaries', len(vx), minimum=5)
bad = tr & (xp | vx)
if bad: pf._fail('split disjoint', f'{len(bad)} binaries in train AND held-out: {sorted(bad)[:5]}')
else: pf._ok('split disjoint', 'train ∩ (xproject ∪ val_xproj) = ∅')
xp_pkgs = {b.split('_')[0] for b in xp | vx}; tr_pkgs = {b.split('_')[0] for b in tr}
leak = xp_pkgs & tr_pkgs
if leak: pf._fail('package disjoint', f'held-out packages also in train: {sorted(leak)}')
else: pf._ok('package disjoint', f'{len(xp_pkgs)} held-out packages, none in train')
mi = json.load(open(cfg['data']['match_index_path']))
pf.count('match_index_v2 records', len(mi), minimum=200000)
have = {r['binary'] for r in mi}
pf.count('train binaries present in index', len(tr & have), minimum=int(0.95 * len(tr)))
vv = json.load(open(cfg['data']['votes_vocab_path']))
pf.count('votes vocab size', vv['vocab_size'], minimum=2000)
if vv.get('built_from', {}).get('tiers') != ['train']: pf._fail('votes vocab provenance', str(vv.get('built_from')))
else: pf._ok('votes vocab provenance', 'built from train tier only')
assert cfg['data'].get('format') == 'v2' and cfg['data'].get('select_on') == 'val_xproj', 'config must be v2 + select_on val_xproj'
pf._ok('config', 'format v2, select_on val_xproj')
pf.done()
PY
python3 -u -m src.training.train --config $CFG --seed $SEED --amp --num-workers 4 --save-name ${NAME}.pt --save-every-epoch
echo "=== training done $(date)"
python3 -u scripts/eval_v2.py --ckpt checkpoints/${NAME}.pt --config $CFG --tiers val_xproj xproject test val_indist \
    --out results/${NAME}_eval.json --amp --num-workers 4 --index-cache results/${NAME}_index.npz
echo "=== eval done $(date)"
