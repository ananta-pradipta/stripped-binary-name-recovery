#!/usr/bin/env python3
"""BLens (retrained on OUR train tier) vs our heads on matched test keys.
BLens predictions come from RunExp's LORD inference log (target:/output: pairs, in the order of the test rows of
xflBlensXProjectData[2]); keys are (binary basename, address). Our heads: R/D from the router features tsv, A4 from the
A4 preds tsv, GBT router refit on val with the same recipe as matched_baselines.py / router2_eval. Same metric-v2
canonicalization (c++filt demangle, template/arg stripping) for every system. Reports overall, by regime, by name stratum."""
import argparse, collections, json, os, pickle, re, subprocess, sys
import numpy as np
sys.path.insert(0, '$WORKSPACE/dh2')
from src.evaluation.metrics import compute_subtoken_f1
WS = '$WORKSPACE/dh2'
ap = argparse.ArgumentParser()
ap.add_argument('--log', required=True, help='LORD-inference-logs-*.txt from the BLens experiment dir')
ap.add_argument('--nlp', default=f'{WS}/blens_ours_v2/xflBlensXProjectData')
ap.add_argument('--feat', default=f'{WS}/results/router_c1soft_l10_interim_features.tsv')
ap.add_argument('--regime', default=f'{WS}/results/c1soft_l10_interim_preds_greedy.tsv')
ap.add_argument('--a4', default=f'{WS}/results/a4_codet5p220m_v1/val_test_symgen_holdout_preds.tsv')
ap.add_argument('--out', default=f'{WS}/results/blens_ours_v2_matched.json')
ap.add_argument('--parse-only', action='store_true')
args = ap.parse_args()

def demangle_many(names):
    todo = sorted({n for n in names if n and n.startswith('_Z')}); out = {}
    for i in range(0, len(todo), 5000):
        ch = todo[i:i+5000]; r = subprocess.run(['c++filt'], input='\n'.join(ch), capture_output=True, text=True)
        out.update(dict(zip(ch, r.stdout.splitlines())))
    return out
def canon(n, dem):
    n = dem.get(n, n); n = re.sub(r'\(.*\)$', '', n); n = re.sub(r'<[^<>]*>', '', n); n = re.sub(r'<[^<>]*>', '', n)
    parts = n.split('::')[-2:] if '::' in n else [n]; return '_'.join(p for p in parts if p)

# ---- BLens log -> per-key prediction
pairs = []; lines = open(args.log, errors='ignore').read().splitlines()
for i, l in enumerate(lines):
    if l.startswith('target: ') and i + 1 < len(lines) and lines[i+1].startswith('output: '):
        pairs.append((l[8:].strip(), lines[i+1][8:].strip()))
nlp = pickle.load(open(args.nlp, 'rb')); test = nlp[2]
print(f'log pairs {len(pairs)} vs test rows {len(test)}')
assert len(pairs) == len(test), 'log/test length mismatch — wrong log or partial inference'
blens = {}; tgt_mismatch = 0; empty = 0
for (t, o), e in zip(pairs, test):
    b = os.path.basename(e[0]); b = b[:-9] if b.endswith('_stripped') else b
    key = ('test', b, int(e[1]))
    if t and t != e[2]: tgt_mismatch += 1
    if not o: empty += 1
    blens[key] = o
print(f'BLens keys {len(blens)}, empty preds {empty}, target/row-name mismatches {tgt_mismatch}, uniq preds {len(set(blens.values()))}')
if args.parse_only: sys.exit(0)

# ---- our heads
FEATS = ['sim1', 'margin', 'ext_jacc', 'str_jacc', 'd_conf', 'd_len', 'n_ext', 'n_str', 'n_blocks']
feat = {}; hdr = None
for l in open(args.feat):
    r = l.rstrip('\n').split('\t')
    if hdr is None: hdr = r; continue
    d = dict(zip(hdr, r))
    if d['bap'].startswith('sub_'):
        a = int(d['bap'][4:], 16)
        for da in (0, 4, -4): feat.setdefault((d['tier'], d['binary'], a + da), d)
reg = {}; hdr = None
for l in open(args.regime):
    r = l.rstrip('\n').split('\t')
    if hdr is None: hdr = r; continue
    d = dict(zip(hdr, r))
    if d['bap_name'].startswith('sub_'):
        a = int(d['bap_name'][4:], 16)
        for da in (0, 4, -4): reg.setdefault((d['tier'], d['binary'], a + da), (d['regime'], d['name_seen']))
a4 = {}
for l in open(args.a4):
    r = l.rstrip('\n').split('\t')
    if r[0] in ('val', 'test'): a4[(r[0], r[1], int(r[2], 16))] = (r[4], float(r[9]) if len(r) > 9 else 0.0)
from sklearn.ensemble import GradientBoostingClassifier
def X(d, a4c): return [float(d[f]) for f in FEATS] + [a4c, float(d['sim1']) - a4c]
val_rows = [(d, *a4[k]) for k, d in feat.items() if k[0] == 'val' and k in a4]
dem = demangle_many([d['true'] for d, _, _ in val_rows] + [d['r_pred'] for d, _, _ in val_rows] + [p for _, p, _ in val_rows])
Xv, yv, wv = [], [], []
for d, p4, c4 in val_rows:
    ct = canon(d['true'], dem); fr = compute_subtoken_f1(canon(d['r_pred'], dem), ct); fa = compute_subtoken_f1(canon(p4, dem), ct) if p4 else 0.0
    Xv.append(X(d, c4)); yv.append(1 if fr >= fa else 0); wv.append(abs(fr - fa) + 1e-3)
gb = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0).fit(np.array(Xv), np.array(yv), sample_weight=np.array(wv))
print('router refit on val rows:', len(val_rows))

# ---- matched rows
rows = []; miss = collections.Counter()
for key, bp in blens.items():
    d = feat.get(key); a = a4.get(key)
    if d is None: miss['no_our_row'] += 1; continue
    if a is None: miss['no_a4'] += 1; continue
    rg = reg.get(key, ('?', '?'))
    rows.append({'key': key, 'pkg': key[1].split('_')[0], 'true': d['true'], 'BLens': bp, 'R': d['r_pred'], 'D': d['d_pred'], 'A4': a[0],
                 'x': X(d, a[1]), 'regime': rg[0], 'seen': rg[1]})
print(f'matched rows {len(rows)} (BLens keys {len(blens)}; missing {dict(miss)}); our test rows {sum(1 for k in feat if k[0]=="test")//3}')
HEADS = ('BLens', 'R', 'D', 'A4')
dem = demangle_many([r['true'] for r in rows] + [r[h] for r in rows for h in HEADS])
pr = gb.predict_proba(np.array([r['x'] for r in rows]))[:, 1]
for r, p in zip(rows, pr):
    ct = canon(r['true'], dem)
    for h in HEADS: r['f_' + h] = compute_subtoken_f1(canon(r[h], dem), ct) if r[h] else 0.0
    r['f_gbt'] = r['f_R'] if p >= 0.5 else r['f_A4']; r['f_oracle'] = max(r['f_R'], r['f_A4'])
def table(rs, label):
    T = {}
    for h in ('BLens', 'R', 'D', 'A4', 'gbt', 'oracle'):
        key = 'f_' + h; g = collections.defaultdict(list)
        for r in rs: g[r['pkg']].append(r[key])
        T[h] = {'micro': round(sum(r[key] for r in rs) / len(rs), 4), 'macro': round(sum(sum(v)/len(v) for v in g.values()) / len(g), 4),
                'EM': round(sum(1 for r in rs if r[key] == 1.0) / len(rs), 4), 'n': len(rs), 'pkgs': len(g)}
    print(f'=== {label} (n={len(rs)}, {len(g)} pkgs) ===')
    for h, v in T.items(): print(f"  {h:<7} micro {v['micro']:.4f} macro {v['macro']:.4f} EM {v['EM']:.4f}")
    return T
report = {'all': table(rows, 'ALL matched test keys')}
for rg in ('FT', 'NCT'):
    sub = [r for r in rows if r['regime'] == rg]
    if sub: report[rg] = table(sub, f'regime {rg}')
for sn, lab in (('1', 'seen_name'), ('0', 'novel_name')):
    sub = [r for r in rows if r['seen'] == sn]
    if sub: report[lab] = table(sub, lab)
# ---- same keys as the SymGen samples (matched_baselines.py), so BLens can sit in one table with SymGen
for sname, mpath in (('FT_sample', f'{WS}/symgen_v2/interim_c_metadata.json'), ('NCT_sample', f'{WS}/symgen_v2/nct_seen_metadata.json')):
    meta = json.load(open(mpath)); keys = {('test', m['binary'], int(m['addr'], 16)) for m in meta}
    sub = [r for r in rows if r['key'] in keys]
    if sub: report[sname] = table(sub, f'{sname} keys (SymGen sample)')
report['coverage'] = {'blens_keys': len(blens), 'matched': len(rows), 'missing': dict(miss), 'empty_blens_preds': empty}
json.dump(report, open(args.out, 'w'), indent=1); print('EFFECT: blens matched scoring done ->', args.out)
