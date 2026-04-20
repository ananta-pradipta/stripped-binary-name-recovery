import re, os, pickle
from collections import defaultdict

LOG = '/project/hz79/_shared/cs785/baselines/blens_user_env/blens_data/xp/ours-cp/COMBO-inference-logs.txt'

def tok(s):
    if not s: return set()
    return set(p.lower() for p in re.split(r'[_\W]', s) if p)

def f1(p, t):
    tp, tg = tok(p), tok(t)
    if not tg or not tp: return 0.0
    inter = tp & tg
    if not inter: return 0.0
    return 2*(len(inter)/len(tp))*(len(inter)/len(tg))/((len(inter)/len(tp))+(len(inter)/len(tg)))

with open('/project/hz79/_shared/cs785/baselines/blens_user_env/blens_data/xflBlensXProjectData', 'rb') as f:
    test = pickle.load(f)[2]
lines = open(LOG).readlines()
pairs = []
for i in range(len(lines)):
    if lines[i].startswith('target: '):
        t = lines[i][8:].strip()
        if i+1 < len(lines) and lines[i+1].startswith('output: '):
            pairs.append((t, lines[i+1][8:].strip()))
assert len(pairs) == len(test)

per = defaultdict(lambda: {'n':0,'sumf1':0.0,'em':0})
for (t, o), e in zip(pairs, test):
    pkg = os.path.basename(e[0]).split('_')[0]
    per[pkg]['n'] += 1
    per[pkg]['sumf1'] += f1(o, t)
    if o == t: per[pkg]['em'] += 1
print(f'{"pkg":<12} {"n":>6} {"F1":>7} {"EM":>7}')
for pkg, s in sorted(per.items(), key=lambda x: -x[1]['sumf1']/x[1]['n']):
    print(f'{pkg:<12} {s["n"]:>6} {s["sumf1"]/s["n"]:>7.3f} {s["em"]/s["n"]:>7.3f}')
