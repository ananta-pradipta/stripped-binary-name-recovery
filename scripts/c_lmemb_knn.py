#!/usr/bin/env python3
"""Tier-C retrieval-head control: kNN over the generation LM's own embeddings.
Answers "why is the retrieval head a contrastive BAP-graph encoder instead of the same LM?":
embed every train/test function by mean-pooling the fine-tuned A4 (CodeT5p-220m modctx) ENCODER
over the same prepared inputs the generation head reads, then top-1 cosine kNN from test to train
and predict the neighbor's name. Output preds.tsv in a4_predict format so score_symgen_full
--extra-head / union tooling can consume it. Compare against the C1 contrastive retrieval head
(test micro 0.1269, NCT 0.5638, novel-EM 0.0002)."""
import json, os, sys
import torch
from torch.utils.data import DataLoader
WS = '$WORKSPACE/dh2'
CKPT = f'{WS}/checkpoints/a4_codet5p220m_modctx_v1/best'
DATA = f'{WS}/results/a4_modctx'
OUT = f'{WS}/results/c_lmemb_knn'
os.makedirs(OUT, exist_ok=True)
MAX_SRC = 1280; BS = 48

sys.path.insert(0, WS)
from src.evaluation.metrics import split_name

def f1_pair(pred, true):
    p = [t.lower() for t in split_name(pred)]; t = [x.lower() for x in split_name(true)]
    if not p or not t: return 0.0
    inter = 0; tt = list(t)
    for x in p:
        if x in tt: tt.remove(x); inter += 1
    if inter == 0: return 0.0
    pr, rc = inter / len(p), inter / len(t)
    return 2 * pr * rc / (pr + rc)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
device = 'cuda'
try:
    tok = AutoTokenizer.from_pretrained(CKPT)
except Exception:
    tok = AutoTokenizer.from_pretrained(f'$WORKSPACE/baselines/hf_local/codet5p-220m')
enc = AutoModelForSeq2SeqLM.from_pretrained(CKPT).get_encoder().to(device).eval()

def load(tier):
    return [json.loads(l) for l in open(f'{DATA}/{tier}.jsonl')]

@torch.no_grad()
def embed(rows, tag):
    E = torch.empty(len(rows), enc.config.d_model, dtype=torch.float16)
    for i in range(0, len(rows), BS):
        batch = [r['code'] for r in rows[i:i+BS]]
        x = tok(batch, max_length=MAX_SRC, truncation=True, padding=True, return_tensors='pt').to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            h = enc(input_ids=x.input_ids, attention_mask=x.attention_mask).last_hidden_state
        m = x.attention_mask.unsqueeze(-1)
        e = (h * m).sum(1) / m.sum(1).clamp(min=1)
        E[i:i+BS] = torch.nn.functional.normalize(e.float(), dim=-1).half().cpu()
        if (i // BS) % 200 == 0: print(f'{tag}: {i}/{len(rows)}', flush=True)
    return E

train = load('train'); test = load('test'); val = load('val')
Etr = embed(train, 'train'); torch.save(Etr, f'{OUT}/train_emb.pt')
names_tr = [r['name'] for r in train]

def knn_predict(rows, E, tier):
    Etr_gpu = Etr.to(device)
    out = []
    for i in range(0, len(rows), 4096):
        sims = (E[i:i+4096].to(device).float() @ Etr_gpu.T.float())
        v, idx = sims.max(dim=1)
        for j, (s, k) in enumerate(zip(v.tolist(), idx.tolist())):
            out.append((rows[i+j], names_tr[k], s))
    return out

report = {}
with open(f'{OUT}/preds.tsv', 'w') as fh:
    fh.write('tier\tbinary\tentry_addr\ttrue\tpred\tregime\tname_seen\tf1_raw\tf1_v2\tconf\n')
    for tier, rows in (('val', val), ('test', test)):
        E = embed(rows, tier); torch.save(E, f'{OUT}/{tier}_emb.pt')
        preds = knn_predict(rows, E, tier)
        f1s = []; ems = []; strata = {}
        for r, pn, s in preds:
            f = f1_pair(pn, r['name']); em = pn == r['name']
            f1s.append(f); ems.append(em)
            seen = bool(r.get('name_seen_in_train', r.get('name_seen')))
            for key in (r.get('regime', '?'), 'seen' if seen else 'novel'):
                strata.setdefault(key, []).append((f, em))
            fh.write(f"{tier}\t{r['binary']}\t{r['addr']}\t{r['name']}\t{pn}\t{r.get('regime','?')}\t{int(seen)}\t{f:.3f}\t{f:.3f}\t{s:.4f}\n")
        rep = {'n': len(f1s), 'micro': sum(f1s)/len(f1s), 'EM': sum(ems)/len(ems)}
        for k, v in strata.items():
            rep[k] = {'micro': sum(a for a,_ in v)/len(v), 'EM': sum(b for _,b in v)/len(v), 'n': len(v)}
        report[tier] = rep
        print(f"EFFECT: lmemb_knn {tier} micro {rep['micro']:.4f} EM {rep['EM']:.4f} " +
              ' '.join(f"{k} {rep[k]['micro']:.4f}" for k in sorted(strata)), flush=True)
json.dump(report, open(f'{OUT}/eval.json', 'w'), indent=1)
print('EFFECT: done', flush=True)
