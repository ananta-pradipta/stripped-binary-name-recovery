#!/usr/bin/env python3
"""A4 generation head: fine-tune a CodeT5+ encoder-decoder on (masked Ghidra decompiled code -> function name).

Input : results/a4_ft/{train,val}.jsonl  (built by scripts/a4_build_ft.py; masking identical to the SymGen baseline input)
Output: checkpoints/a4_<tag>/best/  (HF save_pretrained) + results/a4_<tag>/val_preds.tsv + log lines
Selection: val_xproj sub-token F1 (metric v2 split_name) on greedy decode, evaluated every --eval-every steps.
No test-set access here; test predictions are produced by scripts/a4_predict.py and scored by eval_v2.
"""
import argparse, json, os, random, sys, time, math
import torch
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, '$WORKSPACE/dh2')
from src.evaluation.metrics import split_name

WS = '$WORKSPACE/dh2'

def f1_pair(pred, gold):
    p = [t.lower() for t in split_name(pred)]; g = [t.lower() for t in split_name(gold)]
    if not p or not g: return 0.0
    from collections import Counter
    inter = sum((Counter(p) & Counter(g)).values())
    if inter == 0: return 0.0
    pr, rc = inter / len(p), inter / len(g)
    return 2 * pr * rc / (pr + rc)

class A4Set(Dataset):
    def __init__(self, path, tok, max_src, max_tgt, limit=None, seed=0, extra=(), extra_cap=None):
        self.rows = [json.loads(l) for l in open(path)]
        rng0 = random.Random(seed + 1)
        for ep in extra:
            rows = [json.loads(l) for l in open(ep)]
            if extra_cap:
                by = {}
                for r in rows: by.setdefault(r['package'], []).append(r)
                rows = []
                for pkg, rs in sorted(by.items()):
                    if len(rs) > extra_cap: rng0.shuffle(rs); rs = rs[:extra_cap]
                    rows += rs
            print(f'extra train {ep}: {len(rows)} rows (cap {extra_cap})', flush=True)
            self.rows += rows
        if limit and limit < len(self.rows):
            rng = random.Random(seed); rng.shuffle(self.rows); self.rows = self.rows[:limit]
        self.tok, self.max_src, self.max_tgt = tok, max_src, max_tgt
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        # keep the head of the function (signature + first statements carry most name evidence)
        return r['code'].strip(), ' '.join(t.lower() for t in split_name(r['name'])), r
    def collate(self, batch):
        src = self.tok([b[0] for b in batch], max_length=self.max_src, truncation=True, padding=True, return_tensors='pt')
        tgt = self.tok([b[1] for b in batch], max_length=self.max_tgt, truncation=True, padding=True, return_tensors='pt')
        labels = tgt.input_ids.clone(); labels[labels == self.tok.pad_token_id] = -100
        return src.input_ids, src.attention_mask, labels, [b[2] for b in batch]

@torch.no_grad()
def evaluate(model, tok, loader, device, max_tgt, out_path=None, amp_dtype=torch.bfloat16):
    model.eval(); f1s = []; by_regime = {}; rows_out = []
    for ids, am, labels, metas in loader:
        with torch.autocast('cuda', dtype=amp_dtype):
            gen = model.generate(input_ids=ids.to(device), attention_mask=am.to(device), max_new_tokens=max_tgt, num_beams=1)
        preds = tok.batch_decode(gen, skip_special_tokens=True)
        for p, m in zip(preds, metas):
            pred_name = '_'.join(p.strip().split())
            f = f1_pair(pred_name, m['name']); f1s.append(f)
            by_regime.setdefault(m.get('regime', '?'), []).append(f)
            rows_out.append((m['key'], m['package'], m['name'], pred_name, f))
    model.train()
    if out_path:
        with open(out_path, 'w') as fh:
            for r in rows_out: fh.write('\t'.join(map(str, r)) + '\n')
    micro = sum(f1s) / max(1, len(f1s))
    return micro, {k: sum(v) / len(v) for k, v in by_regime.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Salesforce/codet5p-220m')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--max-src', type=int, default=1024)
    ap.add_argument('--max-tgt', type=int, default=24)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--accum', type=int, default=2)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--epochs', type=float, default=3)
    ap.add_argument('--eval-every', type=int, default=2000)
    ap.add_argument('--val-limit', type=int, default=4000, help='val subset for periodic eval (full val at the end)')
    ap.add_argument('--train-limit', type=int, default=None)
    ap.add_argument('--extra-train', nargs='*', default=[], help='additional jsonl files (e.g. SymGen corpus rows)')
    ap.add_argument('--extra-cap', type=int, default=None, help='max rows per package from extra files (B3 domain balance)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--data-dir', default=f'{WS}/results/a4_ft', help='dir with train.jsonl/val.jsonl (rows: code,name,key,package); e.g. results/a4_baptext for the BAP-text control')
    ap.add_argument('--bf16', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed)
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_cosine_schedule_with_warmup
    device = 'cuda'
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    if args.smoke:
        args.train_limit = args.train_limit or 512; args.val_limit = 256; args.epochs = 1; args.eval_every = 16
    train = A4Set(f'{args.data_dir}/train.jsonl', tok, args.max_src, args.max_tgt, args.train_limit, args.seed,
                  extra=args.extra_train, extra_cap=args.extra_cap)
    val_full = A4Set(f'{args.data_dir}/val.jsonl', tok, args.max_src, args.max_tgt)
    val_sub = A4Set(f'{args.data_dir}/val.jsonl', tok, args.max_src, args.max_tgt, args.val_limit, args.seed)
    tl = DataLoader(train, batch_size=args.bs, shuffle=True, collate_fn=train.collate, num_workers=4, drop_last=True)
    vl_sub = DataLoader(val_sub, batch_size=args.bs * 2, collate_fn=val_sub.collate, num_workers=2)
    vl_full = DataLoader(val_full, batch_size=args.bs * 2, collate_fn=val_full.collate, num_workers=2)
    steps_total = int(len(tl) * args.epochs / args.accum)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sch = get_cosine_schedule_with_warmup(opt, int(0.03 * steps_total), steps_total)
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=not args.bf16)
    ckdir = f'{WS}/checkpoints/a4_{args.tag}'; resdir = f'{WS}/results/a4_{args.tag}'
    os.makedirs(ckdir, exist_ok=True); os.makedirs(resdir, exist_ok=True)
    print(f'A4 {args.tag}: model={args.model} params={nparams/1e6:.1f}M train={len(train)} val={len(val_full)} '
          f'steps={steps_total} bs={args.bs}x{args.accum} max_src={args.max_src}', flush=True)
    best = -1.0; step = 0; t0 = time.time(); model.train(); running = 0.0; n_run = 0
    done = False; ep = 0
    while not done:
        ep += 1
        for i, (ids, am, labels, _) in enumerate(tl):
            with torch.autocast('cuda', dtype=amp_dtype):
                out = model(input_ids=ids.to(device), attention_mask=am.to(device), labels=labels.to(device))
                loss = out.loss / args.accum
            scaler.scale(loss).backward(); running += loss.item() * args.accum; n_run += 1
            if (i + 1) % args.accum == 0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sch.step(); step += 1
                if step % 100 == 0:
                    print(f'step {step}/{steps_total} ep{ep} loss {running/n_run:.4f} lr {sch.get_last_lr()[0]:.2e} '
                          f'{(time.time()-t0)/60:.1f}m', flush=True); running = 0.0; n_run = 0
                if step % args.eval_every == 0 or step >= steps_total:
                    micro, byr = evaluate(model, tok, vl_sub, device, args.max_tgt, amp_dtype=amp_dtype)
                    flag = ''
                    if micro > best:
                        best = micro; model.save_pretrained(f'{ckdir}/best'); tok.save_pretrained(f'{ckdir}/best'); flag = ' *best*'
                    print(f'EVAL step {step}: val_sub F1 {micro:.4f} by_regime {json.dumps({k: round(v,4) for k,v in byr.items()})}{flag}', flush=True)
                if step >= steps_total: done = True; break
    # final: full val with the best checkpoint
    model = AutoModelForSeq2SeqLM.from_pretrained(f'{ckdir}/best').to(device)
    micro, byr = evaluate(model, tok, vl_full, device, args.max_tgt, out_path=f'{resdir}/val_preds.tsv', amp_dtype=amp_dtype)
    json.dump({'val_full_micro': micro, 'by_regime': byr, 'best_sub': best, 'args': vars(args), 'params': nparams},
              open(f'{resdir}/val_summary.json', 'w'), indent=1)
    print(f'EFFECT: a4 {args.tag} FULL val_xproj F1 {micro:.4f} by_regime {json.dumps({k: round(v,4) for k,v in byr.items()})} '
          f'(best sub {best:.4f}) ckpt {ckdir}/best', flush=True)

if __name__ == '__main__':
    main()
