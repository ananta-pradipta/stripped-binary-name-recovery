#!/usr/bin/env python3
"""LoRA contrastive retrieval adapter (user-approved 2026-09-03). Attacks the 0.47 novel-name
selection gap UNDER the pooling bottleneck (the frozen-projection probe showed pooled vectors
don't carry name-relative structure; LoRA changes token mixing before pooling, plus a learned
attention-pooling head replaces mean-pool).
Design: rank-16 LoRA on every encoder self-attention q/v (base weights FROZEN — generation head
bitwise unchanged), + attention pooling (w2·tanh(W1 h)). Loss: soft supervised-contrastive with
weights = name-token F1; batches sampled as name groups (24 groups × 2 members) so same-name
positives are guaranteed. Val-probed every 1000 steps (30K train index / 3K val queries); best
adapter kept; then full re-embed of train/val/test, kNN preds tsv, unified canon scoring, and the
MLP-routed system vs the demangled generation head."""
import csv, json, os, re, subprocess, sys, collections, random
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, '$WORKSPACE/dh2')
from src.evaluation.metrics import compute_subtoken_f1, split_name
WS = '$WORKSPACE/dh2'
OUT = f'{WS}/results/c_lora_contrastive'
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(42); np.random.seed(42); random.seed(42)
dev = 'cuda'
M = '$WORKSPACE/baselines/hf_local/codet5p-220m'
MAX_SRC = 1280; BS_GROUPS = 24; PER_GROUP = 2; STEPS = 6000; TAU = 0.07; LR = 1e-3

def demangle_many(names):
    todo = sorted({n for n in names if n and n.startswith('_Z')}); out = {}
    for i in range(0, len(todo), 5000):
        ch = todo[i:i+5000]; r = subprocess.run(['c++filt'], input='\n'.join(ch), capture_output=True, text=True)
        out.update(dict(zip(ch, r.stdout.splitlines())))
    return out
def canon(n, dem):
    n = dem.get(n, n); n = re.sub(r'\(.*\)$', '', n); n = re.sub(r'<[^<>]*>', '', n); n = re.sub(r'<[^<>]*>', '', n)
    parts = n.split('::')[-2:] if '::' in n else [n]; return '_'.join(p for p in parts if p)

def rows_of(tier): return [json.loads(l) for l in open(f'{WS}/results/a4_modctx/{tier}.jsonl')]
tr_rows, va_rows, te_rows = rows_of('train'), rows_of('val'), rows_of('test')
dem = demangle_many([r['name'] for r in tr_rows + va_rows + te_rows])
tr_canon = [canon(r['name'], dem) for r in tr_rows]
vocab = {}
def tok_ids(cn):
    ids = []
    for t in sorted(set(x.lower() for x in split_name(cn))):
        if t not in vocab: vocab[t] = len(vocab)
        ids.append(vocab[t])
    return ids
tr_tok = [tok_ids(c) for c in tr_canon]
by_name = collections.defaultdict(list)
for i, c in enumerate(tr_canon): by_name[c].append(i)
multi = [v for v in by_name.values() if len(v) >= 2]
print(f'EFFECT: {len(by_name)} unique names, {len(multi)} with >=2 members', flush=True)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tok = AutoTokenizer.from_pretrained(M)
enc = AutoModelForSeq2SeqLM.from_pretrained(M).get_encoder().to(dev)
for p in enc.parameters(): p.requires_grad_(False)

class LoRALinear(nn.Module):
    def __init__(self, base, r=16, alpha=32):
        super().__init__(); self.base = base
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r
    def forward(self, x):
        return self.base(x) + torch.nn.functional.linear(torch.nn.functional.linear(x, self.A), self.B) * self.scale
n_lora = 0
for blk in enc.block:
    sa = blk.layer[0].SelfAttention
    sa.q = LoRALinear(sa.q).to(dev); sa.v = LoRALinear(sa.v).to(dev); n_lora += 2
class AttnPool(nn.Module):
    def __init__(self, d=768, h=256):
        super().__init__(); self.w1 = nn.Linear(d, h); self.w2 = nn.Linear(h, 1)
    def forward(self, H, mask):
        s = self.w2(torch.tanh(self.w1(H))).squeeze(-1)
        s = s.masked_fill(mask == 0, -1e4)
        a = torch.softmax(s, dim=-1)
        return torch.nn.functional.normalize((H * a.unsqueeze(-1)).sum(1), dim=-1)
pool = AttnPool().to(dev)
train_params = [p for p in enc.parameters() if p.requires_grad] + list(pool.parameters())
print(f'EFFECT: {n_lora} LoRA layers, trainable params {sum(p.numel() for p in train_params):,}', flush=True)
opt = torch.optim.AdamW(train_params, lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)

V = len(vocab)
def name_f1_matrix(idx):
    B = torch.zeros(len(idx), V, device=dev)
    for j, i in enumerate(idx):
        if tr_tok[i]: B[j, tr_tok[i]] = 1.0
    inter = B @ B.T; sz = B.sum(1)
    f1 = 2 * inter / (sz[:, None] + sz[None, :]).clamp(min=1)
    f1.fill_diagonal_(0); return f1

def encode_batch(rows_idx, rows_src, grad=True):
    texts = [rows_src[i]['code'].strip() for i in rows_idx]
    x = tok(texts, max_length=MAX_SRC, truncation=True, padding=True, return_tensors='pt').to(dev)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx, torch.autocast('cuda', dtype=torch.bfloat16):
        H = enc(input_ids=x.input_ids, attention_mask=x.attention_mask).last_hidden_state
        Z = pool(H.float(), x.attention_mask)
    return Z

# fixed probe: 30K train index + 3K val queries
probe_tr = random.sample(range(len(tr_rows)), 30000)
probe_va = random.sample(range(len(va_rows)), 3000)
def probe():
    enc.eval(); pool.eval()
    with torch.no_grad():
        idxE = torch.cat([encode_batch(probe_tr[i:i+64], tr_rows, grad=False) for i in range(0, len(probe_tr), 64)])
        qE = torch.cat([encode_batch(probe_va[i:i+64], va_rows, grad=False) for i in range(0, len(probe_va), 64)])
    top = (qE @ idxE.T).argmax(1).tolist()
    f1s = [compute_subtoken_f1(tr_canon[probe_tr[t]], canon(va_rows[q]['name'], dem)) for q, t in zip(probe_va, top)]
    enc.train(); pool.train()
    return sum(f1s)/len(f1s)

best = (-1, None)
for step in range(1, STEPS + 1):
    groups = random.sample(multi, BS_GROUPS)
    idx = [i for g in groups for i in random.sample(g, PER_GROUP)]
    Z = encode_batch(idx, tr_rows, grad=True)
    W = name_f1_matrix(idx)
    logits = (Z @ Z.T) / TAU; logits.fill_diagonal_(-1e4)
    keep = W.sum(1) > 0
    logp = torch.log_softmax(logits[keep], dim=1)
    target = W[keep] / W[keep].sum(1, keepdim=True)
    loss = -(target * logp).sum(1).mean()
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(train_params, 1.0)
    opt.step(); sched.step()
    if step % 100 == 0: print(f'step {step} loss {loss.item():.4f}', flush=True)
    if step % 1000 == 0:
        v = probe()
        print(f'EVAL step {step}: probe {v:.4f}', flush=True)
        if v > best[0]:
            best = (v, ({k: p.detach().clone().cpu() for k, p in enc.state_dict().items() if 'A' == k.split('.')[-1] or 'B' == k.split('.')[-1]},
                        {k: p.detach().clone().cpu() for k, p in pool.state_dict().items()}))
if best[1] is None: print('EFFECT: no checkpoint kept'); sys.exit(1)
sd = enc.state_dict(); sd.update({k: v.to(dev) for k, v in best[1][0].items()}); enc.load_state_dict(sd)
pool.load_state_dict({k: v.to(dev) for k, v in best[1][1].items()})
torch.save({'lora': best[1][0], 'pool': best[1][1]}, f'{OUT}/adapter.pt')
print(f'EFFECT: best probe {best[0]:.4f}, adapter saved', flush=True)

# baseline probe reference for the log: raw mean-pool space probe (same subsets) from saved embs
Eraw_tr = torch.nn.functional.normalize(torch.load(f'{WS}/results/c_lmemb_knn/train_emb.pt', weights_only=True).float(), dim=-1)
Eraw_va = torch.nn.functional.normalize(torch.load(f'{WS}/results/c_lmemb_knn/val_emb.pt', weights_only=True).float(), dim=-1)
topr = (Eraw_va[probe_va].to(dev) @ Eraw_tr[probe_tr].to(dev).T).argmax(1).tolist()
rawp = sum(compute_subtoken_f1(tr_canon[probe_tr[t]], canon(va_rows[q]['name'], dem)) for q, t in zip(probe_va, topr))/len(probe_va)
print(f'EFFECT: RAW-space same-probe reference {rawp:.4f}', flush=True)

# full re-embed + kNN + unified scoring + routed system
enc.eval(); pool.eval()
def embed_all(rows_src, tag):
    Z = torch.empty(len(rows_src), 768)
    for i in range(0, len(rows_src), 64):
        Z[i:i+64] = encode_batch(list(range(i, min(i+64, len(rows_src)))), rows_src, grad=False).cpu()
        if (i // 64) % 300 == 0: print(f'{tag}: {i}/{len(rows_src)}', flush=True)
    return Z
Ztr = embed_all(tr_rows, 'embed-train'); torch.save(Ztr, f'{OUT}/train_emb.pt')
names_tr = [r['name'] for r in tr_rows]
Ztr_g = Ztr.to(dev)
marg = {}
with open(f'{OUT}/preds.tsv', 'w') as fh:
    fh.write('tier\tbinary\tentry_addr\ttrue\tpred\tregime\tname_seen\tf1_raw\tf1_v2\tconf\n')
    for tier, rs in (('val', va_rows), ('test', te_rows)):
        Z = embed_all(rs, f'embed-{tier}'); torch.save(Z, f'{OUT}/{tier}_emb.pt')
        for i in range(0, len(rs), 4096):
            v2, ix = (Z[i:i+4096].to(dev) @ Ztr_g.T).topk(2, dim=1)
            for j in range(v2.shape[0]):
                r = rs[i+j]; seen = bool(r.get('name_seen_in_train', r.get('name_seen')))
                fh.write(f"{tier}\t{r['binary']}\t{r['addr']}\t{r['name']}\t{names_tr[ix[j,0]]}\t{r.get('regime','?')}\t{int(seen)}\t0\t0\t{v2[j,0].item():.4f}\n")
                marg[(tier, r['binary'], r['addr'])] = (float(v2[j,0]), float(v2[j,0]-v2[j,1]))
print('EFFECT: lora preds written', flush=True)
proto = {}
for tier in ('val','test'):
    for l in open(f'{WS}/results/baseline_protocol_v2/{tier}.jsonl'):
        r = json.loads(l); proto[(tier, r['binary'], r['entry_addr'])] = (r['name'], r.get('regime','?'), bool(r.get('name_seen_in_train')))
def load_preds(path):
    d = {}
    with open(path) as fh:
        for x in csv.DictReader(fh, delimiter='\t'):
            if x['tier'] in ('val','test'): d[(x['tier'], x['binary'], x['entry_addr'])] = (x['pred'], float(x['conf']))
    return d
Rp = load_preds(f'{OUT}/preds.tsv'); A = load_preds(f'{WS}/results/a4_codet5p220m_modctx_dm_v1/val_test_preds.tsv')
keys = sorted(set(Rp) & set(A) & set(proto) & set(marg))
dem2 = demangle_many([proto[k][0] for k in keys] + [Rp[k][0] for k in keys] + [A[k][0] for k in keys])
rows = []
for k in keys:
    ct = canon(proto[k][0], dem2)
    fR = compute_subtoken_f1(canon(Rp[k][0], dem2), ct) if Rp[k][0] else 0.0
    fA = compute_subtoken_f1(canon(A[k][0], dem2), ct) if A[k][0] else 0.0
    s, m = marg[k]
    rows.append({'tier': k[0], 'pkg': k[1].split('_')[0], 'regime': proto[k][1], 'seen': proto[k][2],
                 'fR': fR, 'fA': fA, 'X': [s, m, A[k][1], s-A[k][1]]})
val = [r for r in rows if r['tier']=='val']; test = [r for r in rows if r['tier']=='test']
def strat(rs, f):
    o = {}
    for s, sub in (('all', rs), ('seen', [r for r in rs if r['seen']]), ('novel', [r for r in rs if not r['seen']]),
                   ('FT', [r for r in rs if r['regime']=='FT']), ('NCT', [r for r in rs if r['regime']=='NCT'])):
        if sub: o[s] = round(sum(r[f] for r in sub)/len(sub), 4)
    return o
print('EFFECT: LORA retrieval head val', strat(val, 'fR'), flush=True)
print('EFFECT: LORA retrieval head test', strat(test, 'fR'), flush=True)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
Xv = np.array([r['X'] for r in val]); Xt = np.array([r['X'] for r in test])
yv = np.array([1 if r['fR'] >= r['fA'] else 0 for r in val])
wv = np.abs(np.array([r['fR']-r['fA'] for r in val])) + 1e-3
sc = StandardScaler().fit(Xv)
idx2 = np.random.default_rng(42).choice(len(val), size=min(len(val)*3, 60000), p=wv/wv.sum())
mlp = MLPClassifier((64,32), max_iter=800, early_stopping=True, random_state=42).fit(sc.transform(Xv[idx2]), yv[idx2])
for label, rs, X in (('val', val, Xv), ('test', test, Xt)):
    useR = mlp.predict(sc.transform(X)).astype(bool)
    fU = [r['fR'] if u else r['fA'] for r, u in zip(rs, useR)]
    pkg = collections.defaultdict(list)
    for r, v in zip(rs, fU): pkg[r['pkg']].append(v)
    mac = round(sum(sum(v)/len(v) for v in pkg.values())/len(pkg), 4)
    o = round(sum(max(r['fR'], r['fA']) for r in rs)/len(rs), 4)
    print(f'EFFECT: SYSTEM(loraR + dm) {label} micro {round(sum(fU)/len(fU),4)} macro {mac} R_rate {float(np.mean(useR)):.3f} oracle {o}', flush=True)
print('EFFECT: done', flush=True)
