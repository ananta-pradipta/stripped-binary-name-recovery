#!/usr/bin/env python3
"""Rescore frozen prediction dumps under metric v2 (camelCase split + C++ demangling).

GT canonicalization for mangled names: c++filt -> drop argument list/template args ->
namespace::Class::method -> subtokens from all components. Predictions get the same
treatment (they are rarely mangled). Raw predictions are NOT changed.
"""
import json, re, subprocess, sys
from collections import defaultdict
sys.path.insert(0, '.')
from src.evaluation.metrics import compute_subtoken_f1, split_name

def demangle_many(names):
    todo = sorted({n for n in names if n.startswith('_Z')})
    out = {}
    for i in range(0, len(todo), 5000):
        chunk = todo[i:i+5000]
        r = subprocess.run(['c++filt'], input='\n'.join(chunk), capture_output=True, text=True)
        for m, d in zip(chunk, r.stdout.splitlines()):
            out[m] = d
    return out

def canon(name, dem):
    n = dem.get(name, name)
    n = re.sub(r'\(.*\)$', '', n)          # drop argument list
    n = re.sub(r'<[^<>]*>', '', n)         # drop template args (one level, repeat)
    n = re.sub(r'<[^<>]*>', '', n)
    n = n.split('::')[-2:] if '::' in n else [n]   # Class::method -> both parts
    return '_'.join(p for p in n if p)

def score(pairs, dem):
    return [(compute_subtoken_f1(canon(p, dem), canon(t, dem)), canon(p, dem) == canon(t, dem)) for p, t in pairs]

def report(tag, rows_pt, pkgs, regimes=None):
    dem = demangle_many([t for _, t in rows_pt] + [p for p, _ in rows_pt])
    sc = score(rows_pt, dem)
    n = len(sc)
    mic = sum(f for f, _ in sc) / n
    em = sum(e for _, e in sc) / n
    g = defaultdict(list)
    for (f, _), p in zip(sc, pkgs):
        g[p].append(f)
    mac = sum(sum(v) / len(v) for v in g.values()) / len(g)
    line = f"{tag:<26} n={n:>7} microF1 {mic:.4f} EM {em:.4f} macroF1 {mac:.4f}"
    if regimes:
        for r in ('FT', 'NCT'):
            sub = [f for (f, _), rr in zip(sc, regimes) if rr == r]
            if sub:
                line += f" | {r} {sum(sub)/len(sub):.4f}"
    print(line)
    return {p: sum(v)/len(v) for p, v in g.items()}

# ---- our heads (diag dump: full test) ----
rows = [l.rstrip('\n').split('\t') for l in open('results/dualhead_v2/diag_p2_dump.tsv')][1:]
R = [dict(zip(['tier','binary','bap','true','d_pred','d_f1','r_pred','r_f1','sim1','margin'], r)) for r in rows]
for tier in ('test', 'val'):
    T = [x for x in R if x['tier'] == tier]
    pk = [x['binary'].split('_')[0] for x in T]
    print(f'--- {tier} (metric v2) ---')
    report('decoder', [(x['d_pred'], x['true']) for x in T], pk)
    report('retrieval', [(x['r_pred'], x['true']) for x in T], pk)
# ---- SymGen interim C (24 clean FT pkgs sample) ----
sg = json.load(open('results/dualhead_v2/symgen_interim_c_score.json'))
pk = [x['pkg'] for x in sg]
print('--- SymGen interim-C sample (metric v2) ---')
per_sg = report('SymGen(34B)', [(x['pred'], x['true']) for x in sg], pk)
per_r = report('our retrieval (same keys)', [(x['r_pred'] if 'r_pred' in x else '', x['true']) for x in sg], pk) if 'r_pred' in sg[0] else None
# matched: recompute our heads on same keys from diag dump
key = {(x['binary'], x['true']): x for x in R if x['tier'] == 'test'}
match = [(sg_x, key.get((sg_x.get('binary', ''), sg_x['true']))) for sg_x in sg]
mt = [(s, o) for s, o in match if o]
if not mt:  # score_c.json rows lack binary; fall back to name-only join within pkg
    byname = defaultdict(list)
    for x in R:
        if x['tier'] == 'test':
            byname[(x['binary'].split('_')[0], x['true'])].append(x)
    mt = [(s, byname[(s['pkg'], s['true'])][0]) for s in sg if byname.get((s['pkg'], s['true']))]
print(f'matched keys: {len(mt)}')
report('  SymGen (matched)', [(s['pred'], s['true']) for s, _ in mt], [s['pkg'] for s, _ in mt])
report('  our retrieval (matched)', [(o['r_pred'], o['true']) for _, o in mt], [s['pkg'] for s, _ in mt])
report('  our decoder (matched)', [(o['d_pred'], o['true']) for _, o in mt], [s['pkg'] for s, _ in mt])
print('EFFECT: rescore done')
