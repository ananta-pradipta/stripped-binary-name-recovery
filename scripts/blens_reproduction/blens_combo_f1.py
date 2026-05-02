import re
from collections import defaultdict

LOG = '<shared-project-root>/baselines/blens_user_env/blens_data/xp/ours-cp/COMBO-inference-logs.txt'

def tok(s):
    if not s: return set()
    return set(p.lower() for p in re.split(r'[_\W]', s) if p)

def f1(p, t):
    tp, tg = tok(p), tok(t)
    if not tg: return 0.0
    if not tp: return 0.0
    inter = tp & tg
    if not inter: return 0.0
    prec = len(inter)/len(tp); rec = len(inter)/len(tg)
    return 2*prec*rec/(prec+rec)

total = 0; sumf1 = 0.0; em = 0; empty = 0
with open(LOG) as f:
    lines = f.readlines()

pairs = []
for i in range(len(lines)):
    if lines[i].startswith('target: '):
        t = lines[i][8:].strip()
        if i+1 < len(lines) and lines[i+1].startswith('output: '):
            o = lines[i+1][8:].strip()
            pairs.append((t, o))

print(f'Total pairs: {len(pairs)}')
unique = set(o for _, o in pairs); print(f'Unique outputs: {len(unique)}')
non_empty = sum(1 for _, o in pairs if o); print(f'Non-empty outputs: {non_empty} ({non_empty*100.0/len(pairs):.2f}%)')

for t, o in pairs:
    total += 1
    fv = f1(o, t)
    sumf1 += fv
    if o == t: em += 1
    if not o: empty += 1

print(f'Overall F1={sumf1/total:.4f}  EM={em/total:.4f}  (empty={empty})')
