import re, os
from collections import defaultdict, Counter

LOG = '<shared-project-root>/baselines/blens_user_env/blens_data/xp/ours-cp/LORD-inference-logs-test-fixed-59.txt'

def tok(s):
    if not s: return set()
    return set(p.lower() for p in re.split(r'[_\W]', s) if p)

def f1(p, t):
    tp, tg = tok(p), tok(t)
    if not tg or not tp: return 0.0
    inter = tp & tg
    if not inter: return 0.0
    prec = len(inter)/len(tp); rec = len(inter)/len(tg)
    return 2*prec*rec/(prec+rec)

# Need test entries in same order to match per-pkg
import pickle
with open('<shared-project-root>/baselines/blens_user_env/blens_data/xflBlensXProjectData', 'rb') as f:
    nlp = pickle.load(f)
test = nlp[2]

pairs = []
lines = open(LOG).readlines()
for i in range(len(lines)):
    if lines[i].startswith('target: '):
        t = lines[i][8:].strip()
        if i+1 < len(lines) and lines[i+1].startswith('output: '):
            o = lines[i+1][8:].strip()
            pairs.append((t, o))

print(f'Total pairs: {len(pairs)}  vs test entries: {len(test)}')
assert len(pairs) == len(test), f'mismatch {len(pairs)} vs {len(test)}'

# Overall
sumf1 = em = empty = 0
for (t, o), e in zip(pairs, test):
    sumf1 += f1(o, t)
    if o == t: em += 1
    if not o: empty += 1
print(f'Overall LORD F1={sumf1/len(pairs):.4f}  EM={em/len(pairs):.4f}  empty={empty}  uniq={len(set(o for _,o in pairs))}')

# Per-pkg
per = defaultdict(lambda: {'n':0,'sumf1':0.0,'em':0})
for (t, o), e in zip(pairs, test):
    pkg = os.path.basename(e[0]).split('_')[0]
    per[pkg]['n'] += 1
    per[pkg]['sumf1'] += f1(o, t)
    if o == t: per[pkg]['em'] += 1

print(f'\n{"pkg":<12} {"n":>6} {"F1":>7} {"EM":>7}')
for pkg, s in sorted(per.items(), key=lambda x: -x[1]['sumf1']/x[1]['n']):
    print(f'{pkg:<12} {s["n"]:>6} {s["sumf1"]/s["n"]:>7.3f} {s["em"]/s["n"]:>7.3f}')
