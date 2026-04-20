import pickle, os
from collections import Counter
with open('/project/hz79/_shared/cs785/baselines/blens_user_env/blens_data/xflBlensXProjectData', 'rb') as f:
    nlp = pickle.load(f)
train, val, test = nlp[0], nlp[1], nlp[2]

xproj_pkgs = {'nginx118','angie','tengine','recutils','gettext','dash','psmisc','grep','sed'}

def pkg_counts(split, name):
    c = Counter()
    for e in split:
        bn = os.path.basename(e[0])
        c[bn.split('_')[0]] += 1
    print(f'{name} split: {len(split)} fns, {len(c)} pkgs')
    xproj_leak = {p: n for p, n in c.items() if p in xproj_pkgs}
    if xproj_leak:
        print(f'  !! XPROJ LEAK: {xproj_leak}')
    else:
        print(f'  OK: no xproj packages present')
    # top 10
    print(f'  top 10: {c.most_common(10)}')
    return c

train_c = pkg_counts(train, 'TRAIN')
print()
val_c = pkg_counts(val, 'VAL')
print()
test_c = pkg_counts(test, 'TEST')
# Confirm test has ONLY xproj pkgs
non_xproj_in_test = {p: n for p, n in test_c.items() if p not in xproj_pkgs}
print(f'\nnon-xproj pkgs in TEST: {non_xproj_in_test}')
