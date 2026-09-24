#!/usr/bin/env python3
"""Where does the evidence for zero/weak-evidence novel-name functions live, if anywhere?
For the novel-name sample rows whose own decomp text lacks GT tokens, measure GT-subtoken coverage in:
  (a) address-adjacent functions' decomp text (±K neighbors in address order ~ translation-unit locality)
  (b) direct callees/callers (functions this one calls / that call it, via FUN_/sub_ refs in decomp text)
  (c) the whole binary's decomp text pool (upper bound for any within-binary evidence pooling)
  (d) prefix recoverability: is the GT name's FIRST token findable in neighbors (naming-convention signal)?
This sizes the headroom of module-evidence pooling / name propagation / project-lexicon prompts."""
import json, os, re, sys, glob, random, collections
sys.path.insert(0, '/project/hz79/_shared/cs785/dh2')
from src.evaluation.metrics import split_name
WS = '/project/hz79/_shared/cs785/dh2'
K = 10  # address neighbors each side

# novel-name sample: reuse fulltest meta (novel rows), sample per binary
random.seed(0)
meta_all = []
for k in range(34):
    for m in json.load(open(f'{WS}/symgen_v2/fulltest_shards/meta_{k}.json')):
        if not m.get('name_seen'): meta_all.append(m)
sample = random.sample(meta_all, 12000)
by_bin = collections.defaultdict(list)
for m in sample: by_bin[m['binary']].append(m)
print(f'sample {len(sample)} novel rows over {len(by_bin)} binaries')

def canon_tokens(name):
    n = re.sub(r'\(.*\)$', '', name); n = re.sub(r'<[^<>]*>', '', n)
    return [t for t in split_name(n) if len(t) > 1]  # drop 1-char tokens (noise)

CALL_RE = re.compile(r'\b(?:FUN_|sub_)([0-9a-fA-F]{4,})')
stats = collections.Counter(); buckets = {b: collections.Counter() for b in ('self0', 'self_weak')}
agg = {b: collections.defaultdict(float) for b in ('self0', 'self_weak')}
done = 0
for b, rs in by_bin.items():
    p = f'{WS}/symgen_v2/decomp/{b}.json'
    if not os.path.exists(p): continue
    dec = json.load(open(p))
    addrs = sorted(dec.keys(), key=lambda a: int(a, 16))
    idx = {a: i for i, a in enumerate(addrs)}
    lower = {a: dec[a].get('code', '').lower() for a in addrs}
    whole = None
    # reverse call map (callers)
    callers = collections.defaultdict(set)
    for a in addrs:
        for h in CALL_RE.findall(dec[a].get('code', '')):
            callers['0x' + h.lstrip('0x').lower()].add(a)
    for m in rs:
        a = m['addr']
        e = dec.get(a)
        if not e or 'code' not in e or a not in idx: continue
        tt = set(canon_tokens(m['gt_name']))
        if not tt: continue
        own = lower[a].replace(e['ghidra_name'].lower(), ' ', 1)
        self_cov = sum(1 for t in tt if t in own) / len(tt)
        bucket = 'self0' if self_cov == 0 else ('self_weak' if self_cov < 0.5 else None)
        if bucket is None: stats['self_has_evidence'] += 1; continue
        stats[bucket] += 1; done += 1
        i = idx[a]
        neigh = ' '.join(lower[x] for x in addrs[max(0, i-K):i] + addrs[i+1:i+1+K])
        ncov = sum(1 for t in tt if t in neigh) / len(tt)
        # callees: sub_/FUN_ addresses referenced in own code; callers from reverse map
        cs = set()
        for h in CALL_RE.findall(e['code']): cs.add('0x' + h.lstrip('0x').lower())
        cs |= callers.get(a, set())
        cav = ' '.join(lower.get(c, '') for c in list(cs)[:30])
        ccov = sum(1 for t in tt if t in cav) / len(tt) if cav else 0.0
        if whole is None: whole = ' '.join(lower.values())
        wcov = sum(1 for t in tt if t in whole) / len(tt)
        first = canon_tokens(m['gt_name'])[0]
        agg[bucket]['neigh'] += ncov; agg[bucket]['call'] += ccov; agg[bucket]['whole'] += wcov
        agg[bucket]['neigh_any'] += 1 if ncov > 0 else 0
        agg[bucket]['neigh_full'] += 1 if ncov >= 0.999 else 0
        agg[bucket]['call_any'] += 1 if ccov > 0 else 0
        agg[bucket]['whole_any'] += 1 if wcov > 0 else 0
        agg[bucket]['whole_full'] += 1 if wcov >= 0.999 else 0
        agg[bucket]['prefix_in_neigh'] += 1 if first in neigh else 0
        agg[bucket]['prefix_in_whole'] += 1 if (whole and first in whole) else 0

print('bucket counts:', dict(stats), 'analyzed:', done)
out = {'sample': len(sample), 'buckets': dict(stats)}
for b in ('self0', 'self_weak'):
    n = stats[b]
    if not n: continue
    out[b] = {k: round(v / n, 3) for k, v in agg[b].items()}
    print(f'\n== {b} (n={n}) — GT-token coverage beyond the function itself ==')
    for k, v in out[b].items(): print(f'  {k:<16} {v:.3f}')
json.dump(out, open(f'{WS}/results/zero_evidence_census.json', 'w'), indent=1)
print('\nEFFECT: zero_evidence_census ->', f'{WS}/results/zero_evidence_census.json')
