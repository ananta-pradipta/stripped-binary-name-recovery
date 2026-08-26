"""C1 — name-aware contrastive training for the retrieval embedding (docs/C1_NAME_AWARE_CONTRASTIVE_DESIGN.md).

NameAwareBatchSampler : each batch = `anchor_count` anchors, each with one positive (cross-package near-name
    J>=0.5 preferred, else exact-name in another binary) and optionally one hard negative (a kNN neighbour from an
    embedding dump whose name F1 < 0.2), plus random fills.
soft_contrastive_loss : soft-label InfoNCE. Target distribution over the batch P_ij ∝ s_ij * (1 + beta*[pkg_i != pkg_j]),
    s_ij = sub-token F1 between names (metric-v2 split); rows with no positive mass are skipped.
    L = mean_i KL(P_i || softmax_j(cos(z_i,z_j)/tau)).  s_ij -> [n_i==n_j] reduces to the old exact-name NT-Xent.
"""
import random
from collections import defaultdict, Counter
import numpy as np
import torch
from torch.utils.data import Sampler
from src.evaluation.metrics import split_name


def _toks(name):
    return frozenset(t.lower() for t in split_name(name))


def name_f1(a, b):
    """Sub-token F1 between two names (sets; matches metric v2 up to multiplicity)."""
    if a == b:
        return 1.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    p, r = inter / len(b), inter / len(a)
    return 2 * p * r / (p + r)


class NameAwareBatchSampler(Sampler):
    def __init__(self, dataset, train_indices, batch_size=64, anchor_count=16, jacc_min=0.5, df_cap=3000,
                 hard_neg=None, hard_neg_max_f1=0.2, seed=42):
        self.batch_size, self.anchor_count = batch_size, anchor_count
        self.train_indices = list(train_indices)
        self.rng = random.Random(seed)
        S = dataset.samples
        self.pkg = {i: S[i]['binary'].split('_')[0] for i in self.train_indices}
        self.name = {i: S[i]['name'] for i in self.train_indices}
        by_name = defaultdict(list)
        for i in self.train_indices:
            by_name[self.name[i]].append(i)
        toks = {n: _toks(n) for n in by_name}
        df = Counter(t for ts in toks.values() for t in ts)
        inv = defaultdict(set)
        for n, ts in toks.items():
            for t in ts:
                if df[t] <= df_cap:
                    inv[t].add(n)
        name_pkgs = {n: {self.pkg[i] for i in idx} for n, idx in by_name.items()}
        # positive pool per NAME (not per index): near names (J>=jacc_min) held by another package, weight = F1
        self.pos_names = {}
        for n, ts in toks.items():
            cands = set()
            for t in ts:
                cands |= inv.get(t, set())
            cands.discard(n)
            near = []
            for c in cands:
                cs = toks[c]
                j = len(ts & cs) / len(ts | cs)
                if j >= jacc_min and (name_pkgs[c] - name_pkgs[n]):
                    near.append((c, name_f1(ts, cs)))
            self.pos_names[n] = near
        self.by_name = by_name
        self.anchors = [i for i in self.train_indices if self.pos_names[self.name[i]] or len(by_name[self.name[i]]) > 1]
        # hard negatives from an embedding dump: {train_idx: [neighbour train_idx, ...]} with name F1 < hard_neg_max_f1
        self.hard = {}
        if hard_neg is not None:
            for i, nbrs in hard_neg.items():
                bad = [j for j in nbrs if j in self.name and name_f1(_toks(self.name[i]), _toks(self.name[j])) < hard_neg_max_f1]
                if bad:
                    self.hard[i] = bad
        n_near = sum(1 for i in self.anchors if self.pos_names[self.name[i]])
        print(f"NameAwareBatchSampler: anchors {len(self.anchors)}/{len(self.train_indices)} (near-name {n_near}, "
              f"exact-only {len(self.anchors) - n_near}); hard-neg anchors {len(self.hard)}; batch {batch_size} = "
              f"{anchor_count} anchors x (1 pos [+1 hardneg]) + fills")

    def _positive(self, i):
        n = self.name[i]; pkg = self.pkg[i]
        near = self.pos_names[n]
        if near and self.rng.random() < 0.7:
            c, _ = self.rng.choice(near)
            cand = [j for j in self.by_name[c] if self.pkg[j] != pkg] or self.by_name[c]
            return self.rng.choice(cand)
        same = [j for j in self.by_name[n] if j != i]
        if same:
            other = [j for j in same if self.pkg[j] != pkg]
            return self.rng.choice(other or same)
        if near:
            c, _ = self.rng.choice(near)
            return self.rng.choice(self.by_name[c])
        return None

    def __iter__(self):
        anchors = self.anchors[:]; self.rng.shuffle(anchors)
        fills = self.train_indices[:]; self.rng.shuffle(fills)
        fi = 0
        for k in range(0, len(anchors), self.anchor_count):
            batch = []
            for a in anchors[k:k + self.anchor_count]:
                p = self._positive(a)
                if p is None:
                    continue
                batch += [a, p]
                if a in self.hard:
                    batch.append(self.rng.choice(self.hard[a]))
            seen = set(batch)
            while len(batch) < self.batch_size and fi < len(fills):
                j = fills[fi]; fi += 1
                if j not in seen:
                    batch.append(j); seen.add(j)
            if len(batch) >= 4:
                yield batch[:self.batch_size]

    def __len__(self):
        return (len(self.anchors) + self.anchor_count - 1) // self.anchor_count

    @property
    def pair_names(self):
        # legacy interface used by train.py's log line: names with at least one positive
        return [n for n in self.by_name if self.pos_names[n] or len(self.by_name[n]) > 1]


def soft_contrastive_loss(z, names, pkgs, temperature=0.1, beta=1.0, exact_only=False):
    """Soft-label InfoNCE over the batch (see module doc). z: (B, D); names/pkgs: lists of B strings."""
    B = z.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=z.device)
    toks = [_toks(n) for n in names]
    S = torch.zeros(B, B)
    for i in range(B):
        for j in range(i + 1, B):
            s = (1.0 if names[i] == names[j] else 0.0) if exact_only else name_f1(toks[i], toks[j])
            if s > 0:
                w = s * (1.0 + (beta if pkgs[i] != pkgs[j] else 0.0))
                S[i, j] = w; S[j, i] = w
    S = S.to(z.device)
    row_mass = S.sum(1)
    valid = row_mass > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=z.device)
    P = S[valid] / row_mass[valid].unsqueeze(1)
    with torch.autocast(device_type='cuda', enabled=False):
        zf = z.float()
        zn = torch.nn.functional.normalize(zf, dim=1)
        sim = torch.mm(zn, zn.t()) / temperature
        sim = sim.masked_fill(torch.eye(B, dtype=torch.bool, device=z.device), -1e4)   # fp32; -1e4 is safe under any dtype
        logq = sim[valid] - torch.logsumexp(sim[valid], dim=1, keepdim=True)
        return -(P * logq).sum(1).mean()


def load_hard_negatives(knn_npz, train_meta_json, dataset, train_indices, k=10):
    """Map an embedding-dump kNN over TRAIN (train_knn.npz: nbrs indexes into train_meta order) to dataset indices."""
    import json, os
    meta = json.load(open(train_meta_json))
    if os.path.exists(knn_npz):
        nbrs = np.load(knn_npz)['nbrs']
    else:
        # embedding dumps only hold val/test->train neighbours; build train->train top-k blockwise (GPU if available)
        emb_path = os.path.join(os.path.dirname(knn_npz), 'train_emb.npy')
        E = torch.from_numpy(np.load(emb_path).astype(np.float32))
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        E = torch.nn.functional.normalize(E.to(dev), dim=1)
        N = E.shape[0]; kk = min(k + 1, N); out_n = np.zeros((N, kk), dtype=np.int64); bs = 4096
        for s0 in range(0, N, bs):
            sims = E[s0:s0 + bs] @ E.t()
            sims[torch.arange(sims.shape[0]), torch.arange(s0, min(s0 + bs, N))] = -2.0   # drop self
            out_n[s0:s0 + bs] = torch.topk(sims, kk, dim=1).indices.cpu().numpy()
        nbrs = out_n
        try:
            np.savez_compressed(knn_npz, nbrs=nbrs)
        except Exception as e:
            print('WARNING: could not cache train kNN:', e)
        del E
        if dev == 'cuda': torch.cuda.empty_cache()
        print(f"C1: built train->train kNN for {N} rows (k={kk}) from {emb_path}")
    key_to_idx = {}
    for i in train_indices:
        s = dataset.samples[i]; key_to_idx[(s['binary'], s['bap_name'])] = i
    meta_idx = [key_to_idx.get((m['binary'], m['bap'])) for m in meta]
    out = {}
    for r, mi in enumerate(meta_idx):
        if mi is None:
            continue
        out[mi] = [meta_idx[j] for j in nbrs[r][:k] if j < len(meta_idx) and meta_idx[j] is not None and meta_idx[j] != mi]
    return out
