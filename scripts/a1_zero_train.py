#!/usr/bin/env python3
"""A1 zero-training experiments on the embedding dump (HPC, CPU).

E1 string-rerank : rescore top-K retrieval candidates with sim + a*strJacc + b*extJacc
                   (a,b tuned on val), emit top candidate's name.
E2 string-emit   : build name candidates from identifier-shaped tokens in the query's
                   own strings; emit best-scored candidate when retrieval is weak
                   (threshold tuned on val).
Metric v2 scoring (camel fix in metrics; C++ GT demangled via c++filt).
"""
import json, re, subprocess, sys
import numpy as np
sys.path.insert(0, '$WORKSPACE/dh2')
from src.evaluation.metrics import compute_subtoken_f1, split_name
from collections import defaultdict

OUT = '$WORKSPACE/dh2/results/emb_v2'
regime = json.load(open('$WORKSPACE/dh2/data/split_v2.json'))['meta']['test_regime']
train_meta = json.load(open(f'{OUT}/train_meta.json'))
tr_names = [m['name'] for m in train_meta]
tr_strs = [set(t for s in m['strings'] for t in (s if isinstance(s, list) else [s])) if m['strings'] else set() for m in train_meta]
# strings in meta are token lists already (loader pre-tokenized); normalize to sets
def sset(x):
    out = set()
    for s in x or []:
        if isinstance(s, str):
            out.add(s.lower())
    return out
tr_strs = [sset(m['strings']) for m in train_meta]
tr_ext = [set(m['ext']) for m in train_meta]

def demangle(names):
    todo = sorted({n for n in names if n.startswith('_Z')})
    out = {}
    for i in range(0, len(todo), 5000):
        ch = todo[i:i + 5000]
        r = subprocess.run(['c++filt'], input='\n'.join(ch), capture_output=True, text=True)
        out.update(dict(zip(ch, r.stdout.splitlines())))
    return out

def canon_f(dem):
    def canon(n):
        n = dem.get(n, n)
        n = re.sub(r'\(.*\)$', '', n); n = re.sub(r'<[^<>]*>', '', n); n = re.sub(r'<[^<>]*>', '', n)
        parts = n.split('::')[-2:] if '::' in n else [n]
        return '_'.join(p for p in parts if p)
    return canon

IDENT = re.compile(r'[A-Za-z_][A-Za-z0-9_]{3,}')

def load(tier):
    meta = json.load(open(f'{OUT}/{tier}_meta.json'))
    z = np.load(f'{OUT}/{tier}_knn.npz')
    return meta, z['sims'].astype('float32'), z['nbrs']

def jacc(a, b):
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))

def eval_config(meta, sims, nbrs, canon, a, b, emit_thr, use_emit):
    f1s = []
    per_pkg = defaultdict(list)
    for j, m in enumerate(meta):
        qs = sset(m['strings']); qe = set(m['ext'])
        best, bs = None, -1e9
        for k in range(nbrs.shape[1]):
            t = int(nbrs[j, k])
            s = sims[j, k] + a * jacc(qs, tr_strs[t]) + b * jacc(qe, tr_ext[t])
            if s > bs:
                bs, best = s, t
        pred = tr_names[best]
        if use_emit and sims[j, 0] < emit_thr:
            # candidate identifiers from own strings (raw strings are token lists; rejoin unlikely) —
            # use identifier-shaped tokens with >=2 subtokens preferred, longest first
            cands = [t for t in qs if IDENT.fullmatch(t)]
            if cands:
                cands.sort(key=lambda c: (len(split_name(c)) >= 2, len(c)), reverse=True)
                pred = cands[0]
        gt = canon(m['name'])
        f = compute_subtoken_f1(canon(pred), gt)
        f1s.append(f)
        per_pkg[m['binary'].split('_')[0]].append(f)
    mic = sum(f1s) / len(f1s)
    mac = sum(sum(v) / len(v) for v in per_pkg.values()) / len(per_pkg)
    return mic, mac, per_pkg

for tier in ('val', 'test'):
    meta, sims, nbrs = load(tier)
    dem = demangle([m['name'] for m in meta])
    canon = canon_f(dem)
    if tier == 'val':
        base = eval_config(meta, sims, nbrs, canon, 0, 0, 0, False)
        print(f'val baseline top1: micro {base[0]:.4f} macro {base[1]:.4f}')
        best_ab, best_v = (0, 0), base[0]
        for a in (0.0, 0.05, 0.1, 0.2, 0.4):
            for b in (0.0, 0.05, 0.1, 0.2):
                if (a, b) == (0, 0):
                    continue
                v = eval_config(meta, sims, nbrs, canon, a, b, 0, False)
                print(f'  rerank a={a} b={b}: micro {v[0]:.4f} macro {v[1]:.4f}')
                if v[0] > best_v:
                    best_v, best_ab = v[0], (a, b)
        print('BEST rerank on val:', best_ab, best_v)
        best_emit, best_ev = 0.0, best_v
        for thr in (0.3, 0.4, 0.5, 0.6):
            v = eval_config(meta, sims, nbrs, canon, *best_ab, thr, True)
            print(f'  +emit thr={thr}: micro {v[0]:.4f} macro {v[1]:.4f}')
            if v[0] > best_ev:
                best_ev, best_emit = v[0], thr
        print('BEST emit thr on val:', best_emit, best_ev)
        CFG = (best_ab, best_emit)
    else:
        (a, b), thr = CFG
        base = eval_config(meta, sims, nbrs, canon, 0, 0, 0, False)
        rr = eval_config(meta, sims, nbrs, canon, a, b, 0, False)
        em = eval_config(meta, sims, nbrs, canon, a, b, thr, thr > 0)
        reg_of = lambda m: regime.get(m['binary'].split('_')[0], '?')
        def by_reg(cfgargs):
            mic, mac, pp = cfgargs
            return mic, mac
        print(f'TEST baseline   micro {base[0]:.4f} macro {base[1]:.4f}')
        print(f'TEST rerank     micro {rr[0]:.4f} macro {rr[1]:.4f}  (a={a}, b={b})')
        print(f'TEST rerank+emit micro {em[0]:.4f} macro {em[1]:.4f} (thr={thr})')
print('EFFECT: a1 zero-train done')
