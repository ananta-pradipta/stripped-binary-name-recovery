#!/bin/bash
#SBATCH --job-name=cs785-eval --account=hz79 --partition=general --qos=standard
#SBATCH --cpus-per-task=4 --mem=48G --time=06:00:00
#SBATCH --output=/project/hz79/_shared/cs785/dh2/slurm/ctx_build_%j.out
set -uo pipefail
cd /project/hz79/_shared/cs785/dh2
source /home/adp232/phd-research-gpu-env/bin/activate
which c++filt || { echo "FATAL: no c++filt"; exit 1; }
# (a) reproduction check: win10 must equal the adopted modctx digests (val+test)
python3 -u scripts/ctx_checks/a4_build_ctxvar.py --modes win10 --tiers val,test || { echo "FATAL: win10 reproduction failed"; exit 1; }
# (b) trainable variants
python3 -u scripts/ctx_checks/a4_build_ctxvar.py --modes callgraph,random --tiers train,val,test || { echo "FATAL: callgraph/random build failed"; exit 1; }
# (c) inference-only variants
python3 -u scripts/ctx_checks/a4_build_ctxvar.py --modes win5,win20,empty --tiers val,test || { echo "FATAL: win5/win20/empty build failed"; exit 1; }
# (d) layout for the TU-locality analysis
python3 -u scripts/ctx_checks/ctx_layout_extract.py results/ctx_layout/test_layout.json || { echo "FATAL: layout extract failed"; exit 1; }
python3 - <<'PY'
import json, sys
ok = True
for m, tiers in (('callgraph', ('train','val','test')), ('random', ('train','val','test')), ('win5', ('val','test')), ('win20', ('val','test')), ('empty', ('val','test')), ('win10', ('val','test'))):
    st = json.load(open(f'results/a4_ctx_{m}_dm/stats.json'))
    ref = {'train': 190133, 'val': 10617, 'test': 268136}
    for t in tiers:
        if st[t]['rows_out'] != ref[t]: print('SMOKE FAIL', m, t, st[t]['rows_out'], '!=', ref[t]); ok = False
    if m == 'test' or 'test' in tiers:
        c = st['test']['targets_canonized']
        if not (30000 < c < 60000): print('SMOKE FAIL canonized', m, c); ok = False
print('SMOKE', 'PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
PY
rc=$?; echo "EFFECT: ctx_build rc=$rc"; exit $rc
