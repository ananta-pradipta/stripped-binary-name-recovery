#!/usr/bin/env python3
"""C1 census: name-aware contrastive positive availability.
For every train function, is there a function in a DIFFERENT package whose name is
(a) identical, (b) sub-token Jaccard >= 0.5, (c) shares >= 1 informative sub-token?
For val/test functions: same questions against TRAIN names (= what a name-aware
embedding could hope to pull in). Uses src.evaluation.metrics.split_name (metric v2)."""
import json, sys, os
from collections import Counter, defaultdict
sys.path.insert(0, '/project/hz79/_shared/cs785/dh2')
from src.evaluation.metrics import split_name
PROTO = '/project/hz79/_shared/cs785/dh2/results/baseline_protocol_v2'
OUT = '/project/hz79/_shared/cs785/dh2/results/c1_census'
os.makedirs(OUT, exist_ok=True)
DF_CAP = 3000   # sub-tokens occurring in more unique names than this are not used for candidate generation

def load(tier):
    return [json.loads(l) for l in open(f'{PROTO}/{tier}.jsonl')]

train = load('train')
name_pkgs = defaultdict(set); name_cnt = Counter()
for r in train:
    name_pkgs[r['name']].add(r['package']); name_cnt[r['name']] += 1
toks = {n: frozenset(t.lower() for t in split_name(n)) for n in name_pkgs}
df = Counter()
for n, ts in toks.items():
    for t in ts: df[t] += 1
inv = defaultdict(set)
for n, ts in toks.items():
    for t in ts:
        if df[t] <= DF_CAP: inv[t].add(n)
print(f'unique train names={len(name_pkgs)}; sub-tokens={len(df)}; capped tokens (df>{DF_CAP}): '
      f'{[t for t,c in df.most_common(40) if c > DF_CAP]}', flush=True)

def best_xpkg(name, pkg, ts, exclude_self):
    """max Jaccard to any train name held by a package != pkg (name itself allowed only if it lives in another pkg)."""
    best = 0.0; best_n = None
    if name in name_pkgs and (name_pkgs[name] - {pkg}):
        return 1.0, name
    cands = set()
    for t in ts:
        if t in inv: cands |= inv[t]
    for c in cands:
        if c == name: continue
        if not (name_pkgs[c] - {pkg}): continue
        cs = toks[c]; j = len(ts & cs) / len(ts | cs)
        if j > best: best, best_n = j, c
    return best, best_n

report = {}
for tier in ['train', 'val', 'test']:
    rows = train if tier == 'train' else load(tier)
    cnt = Counter(); by_regime = defaultdict(Counter); examples = []
    memo = {}
    for r in rows:
        key = (r['name'], r['package'])
        if key not in memo:
            ts = toks.get(r['name']) or frozenset(t.lower() for t in split_name(r['name']))
            memo[key] = best_xpkg(r['name'], r['package'], ts, tier == 'train')
        j, m = memo[key]
        bucket = 'exact' if j >= 1.0 else 'j>=0.5' if j >= 0.5 else 'j>0' if j > 0 else 'none'
        cnt[bucket] += 1; cnt['total'] += 1
        if tier != 'train':
            by_regime[r.get('regime', '?')][bucket] += 1; by_regime[r.get('regime', '?')]['total'] += 1
        if bucket == 'j>=0.5' and len(examples) < 25 and j < 1.0:
            examples.append((r['package'], r['name'], m, round(j, 2)))
    tot = cnt['total']
    frac = {k: round(cnt[k]/tot, 4) for k in ['exact', 'j>=0.5', 'j>0', 'none']}
    report[tier] = {'counts': dict(cnt), 'frac': frac,
                    'by_regime': {k: {kk: round(v[kk]/v['total'], 4) for kk in ['exact', 'j>=0.5', 'j>0', 'none']}
                                  for k, v in by_regime.items()}, 'examples': examples}
    print(f'EFFECT: c1 {tier}: n={tot} exact-xpkg={frac["exact"]:.3f} near(J>=.5)={frac["j>=0.5"]:.3f} '
          f'weak(J>0)={frac["j>0"]:.3f} none={frac["none"]:.3f}  by_regime={report[tier]["by_regime"]}', flush=True)
json.dump(report, open(f'{OUT}/report.json', 'w'), indent=1)
