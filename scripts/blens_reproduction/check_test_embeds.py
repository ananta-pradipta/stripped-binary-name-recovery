import pickle, os
from collections import Counter

for name in ['clap_test', 'palmtree_test']:
    p = f'<shared-project-root>/baselines/blens_user_env/blens_data/embedding/{name}'
    with open(p, 'rb') as f: d = pickle.load(f)
    print(f'{name}: {len(d)} keys')
    pkg_ct = Counter()
    for k in d.keys():
        bn = k[0] if isinstance(k, tuple) else k
        bn = os.path.basename(bn) if '/' in bn else bn
        pkg = bn.split('_')[0]
        pkg_ct[pkg] += 1
    for pkg, n in pkg_ct.most_common():
        print(f'  {pkg:<14}: {n}')
    # Check xproj test lookup
    with open('<shared-project-root>/baselines/blens_user_env/blens_data/xflBlensXProjectData', 'rb') as f:
        nlp = pickle.load(f)
    test = nlp[2]
    raw = bn_hit = 0
    for e in test:
        bp, addr = e[0], e[1]
        if (bp, addr) in d: raw += 1
        if (os.path.basename(bp), addr) in d: bn_hit += 1
    print(f'  test lookup raw: {raw}/{len(test)} ({raw*100/len(test):.1f}%)')
    print(f'  test lookup basename: {bn_hit}/{len(test)} ({bn_hit*100/len(test):.1f}%)')
    print()
