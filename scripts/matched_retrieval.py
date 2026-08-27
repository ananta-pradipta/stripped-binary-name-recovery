#!/usr/bin/env python3
"""Population-matched retrieval comparison between two embedding dumps (kNN top-1 name, metric v2 F1 with demangle):
scores each dump on the test rows common to both (key = binary, bap_name). Usage: matched_retrieval.py emb_A emb_B"""
import json, sys, re, subprocess, collections
import numpy as np
sys.path.insert(0, '/project/hz79/_shared/cs785/dh2')
from src.evaluation.metrics import compute_subtoken_f1
WS = '/project/hz79/_shared/cs785/dh2'
def demangle_many(names):
    todo = sorted({n for n in names if n and n.startswith('_Z')}); out = {}
    for i in range(0, len(todo), 5000):
        ch = todo[i:i+5000]; r = subprocess.run(['c++filt'], input='\n'.join(ch), capture_output=True, text=True)
        out.update(dict(zip(ch, r.stdout.splitlines())))
    return out
def canon(n, dem):
    n = dem.get(n, n); n = re.sub(r'\(.*\)$', '', n); n = re.sub(r'<[^<>]*>', '', n); n = re.sub(r'<[^<>]*>', '', n)
    parts = n.split('::')[-2:] if '::' in n else [n]; return '_'.join(p for p in parts if p)
regime = {}
for split in ('v2', 'v3'):
    try:
        m = json.load(open(f'{WS}/data/split_{split}.json'))['meta']['test_regime']; regime.update(m)
    except Exception: pass
def load(d):
    tm = json.load(open(f'{d}/train_meta.json')); te = json.load(open(f'{d}/test_meta.json')); z = np.load(f'{d}/test_knn.npz')
    top = z['nbrs'][:, 0]
    return {(m['binary'], m['bap']): (m['name'], tm[int(t)]['name']) for m, t in zip(te, top)}
A, B = load(sys.argv[1]), load(sys.argv[2])
common = sorted(set(A) & set(B)); print(f'{sys.argv[1]}: {len(A)} rows | {sys.argv[2]}: {len(B)} rows | common {len(common)}')
dem = demangle_many([A[k][0] for k in common] + [A[k][1] for k in common] + [B[k][1] for k in common])
def agg(D, tag):
    f = []; g = collections.defaultdict(list); ft = []; nct = []
    for k in common:
        t, p = D[k]; s = compute_subtoken_f1(canon(p, dem), canon(t, dem)); pkg = k[0].split('_')[0]
        f.append(s); g[pkg].append(s); (ft if regime.get(pkg) == 'FT' else nct if regime.get(pkg) == 'NCT' else []).append(s)
    print(f'  {tag:<40} micro {sum(f)/len(f):.4f} macro {sum(sum(v)/len(v) for v in g.values())/len(g):.4f} FT {sum(ft)/max(1,len(ft)):.4f} NCT {sum(nct)/max(1,len(nct)):.4f} pkgs {len(g)}')
agg(A, sys.argv[1].split('/')[-1]); agg(B, sys.argv[2].split('/')[-1])
print('EFFECT: matched retrieval done')
