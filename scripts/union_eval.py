#!/usr/bin/env python3
"""Head-union table: BAP retrieval (r_pred) / BAP decoder (d_pred) from router features TSV  +  A4 generation head preds.
Scores every head with the same metric-v2 canonicalization (demangle + camel split). Routers:
  oracle       : per-row max over heads (upper bound)
  regime       : NCT package -> retrieval, FT package -> A4
  conf(sim1)   : retrieval if sim1 >= tau else A4   (tau swept on val, applied to test)
  conf(margin) : retrieval if margin >= tau else A4
Join key: (binary, entry addr) with bap sub_XXXX -> addr, tolerating +-4 (ENDBR64), fallback (binary, true name).
"""
import json, re, subprocess, sys, collections, argparse
sys.path.insert(0, '/project/hz79/_shared/cs785/dh2')
from src.evaluation.metrics import compute_subtoken_f1
WS = '/project/hz79/_shared/cs785/dh2'
ap = argparse.ArgumentParser(); ap.add_argument('--router', default=f'{WS}/results/router_p3a_features.tsv')
ap.add_argument('--a4', default=f'{WS}/results/a4_codet5p220m_v1/val_test_symgen_holdout_preds.tsv')
ap.add_argument('--regime-dump', default=f'{WS}/results/p3a_preds_greedy.tsv')
ap.add_argument('--out', default=f'{WS}/results/union_a1a_a4v1.json'); args = ap.parse_args()

def demangle_many(names):
    todo = sorted({n for n in names if n and n.startswith('_Z')}); out = {}
    for i in range(0, len(todo), 5000):
        ch = todo[i:i+5000]; r = subprocess.run(['c++filt'], input='\n'.join(ch), capture_output=True, text=True)
        out.update(dict(zip(ch, r.stdout.splitlines())))
    return out
def canon(n, dem):
    n = dem.get(n, n); n = re.sub(r'\(.*\)$', '', n); n = re.sub(r'<[^<>]*>', '', n); n = re.sub(r'<[^<>]*>', '', n)
    parts = n.split('::')[-2:] if '::' in n else [n]; return '_'.join(p for p in parts if p)

# regime per (tier, binary, bap) from eval_v2 dump
regime = {}
for l in open(args.regime_dump):
    r = l.rstrip('\n').split('\t')
    if r[0] in ('val', 'test'): regime[(r[0], r[1], r[2])] = r[5]
# A4 preds
a4_addr, a4_name = {}, {}
for l in open(args.a4):
    r = l.rstrip('\n').split('\t')
    if r[0] not in ('val', 'test'): continue
    a4_addr[(r[0], r[1], int(r[2], 16))] = r[4]; a4_name.setdefault((r[0], r[1], r[3]), r[4])
rows = []
hdr = None; miss = collections.Counter()
for l in open(args.router):
    r = l.rstrip('\n').split('\t')
    if hdr is None: hdr = r; continue
    d = dict(zip(hdr, r)); tier = d['tier']
    if tier not in ('val', 'test'): continue
    a = int(d['bap'][4:], 16) if d['bap'].startswith('sub_') else None
    p4 = None
    if a is not None:
        for da in (0, -4, 4):
            p4 = a4_addr.get((tier, d['binary'], a + da))
            if p4 is not None: miss['addr' if da == 0 else 'addr+-4'] += 1; break
    if p4 is None:
        p4 = a4_name.get((tier, d['binary'], d['true']))
        miss['byname' if p4 is not None else 'MISSING'] += 1
    rows.append({'tier': tier, 'pkg': d['binary'].split('_')[0], 'true': d['true'], 'R': d['r_pred'], 'D': d['d_pred'], 'A4': p4 or '',
                 'sim1': float(d['sim1']), 'margin': float(d['margin']), 'regime': regime.get((tier, d['binary'], d['bap']), '?')})
print('join:', dict(miss), 'rows', len(rows))
dem = demangle_many([x['true'] for x in rows] + [x[h] for x in rows for h in ('R', 'D', 'A4')])
for x in rows:
    ct = canon(x['true'], dem)
    for h in ('R', 'D', 'A4'): x['f_' + h] = compute_subtoken_f1(canon(x[h], dem), ct) if x[h] else 0.0
def agg(rs, key):
    n = len(rs); mic = sum(r[key] for r in rs) / n if n else 0
    g = collections.defaultdict(list)
    for r in rs: g[r['pkg']].append(r[key])
    mac = sum(sum(v)/len(v) for v in g.values()) / len(g) if g else 0
    ft = [r[key] for r in rs if r['regime'] == 'FT']; nct = [r[key] for r in rs if r['regime'] == 'NCT']
    return {'n': n, 'micro': round(mic, 4), 'macro': round(mac, 4), 'FT': round(sum(ft)/len(ft), 4) if ft else None, 'NCT': round(sum(nct)/len(nct), 4) if nct else None}
def apply(rs, choose):
    for r in rs: r['f_U'] = r['f_R'] if choose(r) else r['f_A4']
    return agg(rs, 'f_U')
report = {}
for tier in ('val', 'test'):
    rs = [r for r in rows if r['tier'] == tier]; T = {}
    T['retrieval'] = agg(rs, 'f_R'); T['decoder'] = agg(rs, 'f_D'); T['A4'] = agg(rs, 'f_A4')
    for r in rs: r['f_O'] = max(r['f_R'], r['f_A4']); r['f_O3'] = max(r['f_R'], r['f_D'], r['f_A4'])
    T['oracle_R_A4'] = agg(rs, 'f_O'); T['oracle_R_D_A4'] = agg(rs, 'f_O3')
    T['regime_router'] = apply(rs, lambda r: r['regime'] == 'NCT')
    report[tier] = T
# confidence routers: sweep tau on val, apply to test
val = [r for r in rows if r['tier'] == 'val']; test = [r for r in rows if r['tier'] == 'test']
for feat in ('sim1', 'margin'):
    vals = sorted({round(r[feat], 3) for r in val}); cand = vals[::max(1, len(vals)//60)]
    best = max(cand, key=lambda t: apply(val, lambda r, t=t: r[feat] >= t)['micro'])
    report['val'][f'conf_{feat}'] = dict(apply(val, lambda r: r[feat] >= best), tau=best)
    report['test'][f'conf_{feat}'] = dict(apply(test, lambda r: r[feat] >= best), tau=best)
    # macro-selected tau as well
    bestm = max(cand, key=lambda t: apply(val, lambda r, t=t: r[feat] >= t)['macro'])
    report['test'][f'conf_{feat}_macroTau'] = dict(apply(test, lambda r: r[feat] >= bestm), tau=bestm)
json.dump(report, open(args.out, 'w'), indent=1)
for tier in ('val', 'test'):
    print(f'=== {tier} ===')
    for k, v in report[tier].items(): print(f"  {k:<22} micro {v['micro']:.4f} macro {v['macro']:.4f} FT {v['FT']} NCT {v['NCT']}" + (f"  tau={v['tau']}" if 'tau' in v else ''))
print('EFFECT: union eval done ->', args.out)
