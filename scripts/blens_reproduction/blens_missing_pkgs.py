import pickle
from collections import Counter
with open('<shared-project-root>/baselines/blens_user_env/blens_data/embedding/clap', 'rb') as f:
    clap = pickle.load(f)
# What packages have CLAP embeddings?
pkg_ct = Counter()
for k in clap.keys():
    binary_name = k[0] if isinstance(k, tuple) else k
    pkg = binary_name.split('_')[0]
    pkg_ct[pkg] += 1
print(f'Total CLAP keys: {len(clap)}  across {len(pkg_ct)} pkgs')
print('Packages with CLAP embeddings:')
for pkg, n in pkg_ct.most_common():
    print(f'  {pkg:<14}: {n}')
