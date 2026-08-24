#!/usr/bin/env python3
"""Score SymGen predictions under the v2 protocol, matched-key against our heads."""
import json, os, sys, re
sys.path.insert(0, '/project/hz79/_shared/cs785/dh2')
from src.evaluation.metrics import compute_subtoken_f1
from collections import defaultdict

WS = '/project/hz79/_shared/cs785/dh2/symgen_v2'
preds = json.load(open(f'{WS}/results_c/predicted_function_name.json'))
meta = json.load(open(f'{WS}/interim_c_metadata.json'))
assert len(preds) <= len(meta), (len(preds), len(meta))

def clean(p):
    if not isinstance(p, str):
        return ''
    p = p.strip().strip('`"\' .')
    m = re.search(r'[A-Za-z_][A-Za-z0-9_]*', p)
    return m.group(0) if m else ''

ours = {}
with open('/project/hz79/_shared/cs785/dh2/results/router_v2_features.tsv') as fh:
    hdr = fh.readline().rstrip('\n').split('\t')
    for line in fh:
        d = dict(zip(hdr, line.rstrip('\n').split('\t')))
        ours[(d['binary'], d['bap'])] = (float(d['r_f1']), float(d['d_f1']))
# map metadata addr -> bap name via key: our features key on bap_name, metadata on addr.
# join through binary+gt_name+uniqueness fallback: use (binary, gt_name).
ours_by_name = {}
with open('/project/hz79/_shared/cs785/dh2/results/router_v2_features.tsv') as fh:
    hdr = fh.readline().rstrip('\n').split('\t')
    for line in fh:
        d = dict(zip(hdr, line.rstrip('\n').split('\t')))
        ours_by_name[(d['binary'], d['true'])] = (float(d['r_f1']), float(d['d_f1']))

rows = []
for p, m in zip(preds, meta):
    raw = p if isinstance(p, str) else (p.get('predicted_name') or p.get('predicted') or '') if isinstance(p, dict) else ''
    raw = raw.replace('The predicted function name is', ' ').replace('</s>', ' ')
    sp = clean(raw)
    if isinstance(p, dict) and p.get('ground_truth') not in (None, m['gt_name']):
        raise AssertionError(f"positional join broken at {m['key']}")
    f1 = compute_subtoken_f1(sp, m['gt_name'])
    o = ours_by_name.get((m['binary'], m['gt_name']))
    rows.append({'pkg': m['package'], 'sg_f1': f1, 'sg_em': float(sp == m['gt_name']),
                 'r_f1': o[0] if o else None, 'd_f1': o[1] if o else None, 'pred': sp, 'true': m['gt_name']})
n = len(rows)
matched = [r for r in rows if r['r_f1'] is not None]
def avg(xs): return sum(xs) / max(1, len(xs))
print(f'scored {n} (matched to ours: {len(matched)})')
print(f"SymGen  micro F1 {avg([r['sg_f1'] for r in rows]):.4f} EM {avg([r['sg_em'] for r in rows]):.4f}")
print(f"matched-key: SymGen {avg([r['sg_f1'] for r in matched]):.4f} | our retrieval {avg([r['r_f1'] for r in matched]):.4f} | our decoder {avg([r['d_f1'] for r in matched]):.4f}")
g = defaultdict(list)
for r in rows:
    g[r['pkg']].append(r)
print(f"macro-pkg: SymGen {avg([avg([x['sg_f1'] for x in v]) for v in g.values()]):.4f}")
print(f"{'pkg':<12} {'n':>5} {'SymGen':>8} {'ourR':>8} {'ourD':>8}")
for p, v in sorted(g.items()):
    mv = [x for x in v if x['r_f1'] is not None]
    print(f"{p:<12} {len(v):>5} {avg([x['sg_f1'] for x in v]):>8.4f} {avg([x['r_f1'] for x in mv]) if mv else -1:>8.4f} {avg([x['d_f1'] for x in mv]) if mv else -1:>8.4f}")
json.dump(rows, open(f'{WS}/score_c.json', 'w'))
print('EFFECT: score done')
