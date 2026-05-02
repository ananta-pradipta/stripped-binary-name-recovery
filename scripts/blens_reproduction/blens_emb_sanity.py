"""Check CLAP + PalmTree embedding lookup for test samples vs train/val.
Test should have same-shape non-zero embeddings, matching keys against xflBlensXProjectData."""
import os, sys, pickle
sys.path.insert(0, '<shared-project-root>/baselines/blens')

DATA_DIR = '<shared-project-root>/baselines/blens_user_env/blens_data'

# Load the xproj data (train/val/test)
with open(os.path.join(DATA_DIR, 'xflBlensXProjectData'), 'rb') as f:
    data = pickle.load(f)
train, val, test = data[0], data[1], data[2]

# Load embeddings
with open(os.path.join(DATA_DIR, 'embedding', 'clap'), 'rb') as f:
    clap = pickle.load(f)
with open(os.path.join(DATA_DIR, 'embedding', 'palmtree'), 'rb') as f:
    palm = pickle.load(f)

print(f'CLAP total keys: {len(clap)}')
print(f'PalmTree total keys: {len(palm)}')

# Sample a few keys from CLAP to understand format
clap_keys = list(clap.keys())
print(f'CLAP sample keys:')
for k in clap_keys[:3]:
    print(f'  key={k}')
    v = clap[k]
    import numpy as np
    arr = np.array(v) if not hasattr(v, 'shape') else v
    print(f'    shape={arr.shape}  mean={arr.mean():.4f}  std={arr.std():.4f}  isfinite={np.isfinite(arr).all()}')

# Check lookup success rate for train/val/test
def lookup_rate(entries, emb_dict, name):
    hit = 0; miss_keys = []
    for e in entries[:5000]:
        bp, addr = e[0], e[1]
        # Try multiple key formats
        k1 = (bp, addr)
        k2 = (os.path.basename(bp), addr)
        if k1 in emb_dict:
            hit += 1
        elif k2 in emb_dict:
            hit += 1
        else:
            if len(miss_keys) < 3: miss_keys.append((bp, addr))
    print(f'{name} lookup (first 5000): {hit}/{min(5000, len(entries))} hit ({hit*100/min(5000,len(entries)):.1f}%)')
    for k in miss_keys: print(f'  MISS: {k}')

print('--- CLAP lookup ---')
lookup_rate(train, clap, 'train')
lookup_rate(val, clap, 'val')
lookup_rate(test, clap, 'test')

print('--- PalmTree lookup ---')
lookup_rate(train, palm, 'train')
lookup_rate(val, palm, 'val')
lookup_rate(test, palm, 'test')

# Check if test CLAP embeddings are non-degenerate
import numpy as np
sample_test_embs = []
for e in test[:500]:
    bp, addr = e[0], e[1]
    for k in [(bp, addr), (os.path.basename(bp), addr)]:
        if k in clap:
            sample_test_embs.append(np.array(clap[k]).flatten())
            break
if sample_test_embs:
    sample_test_embs = np.stack(sample_test_embs)
    print(f'test CLAP embeddings (n={len(sample_test_embs)}):')
    print(f'  mean magnitude: {np.abs(sample_test_embs).mean():.4f}')
    print(f'  pairwise cosine between first 5:')
    from numpy.linalg import norm
    normed = sample_test_embs[:5] / (norm(sample_test_embs[:5], axis=1, keepdims=True) + 1e-8)
    cos = normed @ normed.T
    print(cos)
