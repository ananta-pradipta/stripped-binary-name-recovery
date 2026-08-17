#!/usr/bin/env python3
"""Dataset-v2 evaluation: decoder head, k-NN retrieval head, oracle, simple gate,
per tier / package / regime / stratum metrics, and per-function records for
risk–coverage analysis.

  python3 scripts/eval_v2.py --ckpt checkpoints/dh2_large_s42.pt --config configs/dualhead_v2_large.yaml \
      --tiers val_xproj xproject test --out results/dh2/eval_s42.json [--amp] [--k 20]

Strata (per function):
  in_dynsym          name visible in the stripped ELF (symbol-visible) — reported separately,
                     EXCLUDED from the headline target set
  seen_name          canonical name (or alias) occurs as a train-tier name
  novel_composable   not seen, but every Votes sub-token is in the train name sub-token set
  oov                at least one sub-token unseen in train names
Heads: `dec` (beam-5), `knn` (top-1 cosine neighbour's name), `oracle` (better of the two),
       `gate` (knn if top-1 sim >= tau else dec; tau tuned on val_xproj if present).
Metrics: sub-token F1 (micro over functions), EM, per-package macro-F1; risk-coverage AURC
using confidence = knn top-1 sim (knn head) / mean beam log-prob (dec head).
"""
import argparse, json, os, sys, time
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
import yaml
from src.models.function_namer import FunctionNamer
from src.preprocessing.dataset_v2 import FunctionDatasetV2
from src.preprocessing.build_votes import VotesTokenizer, split_name_to_subtokens
from src.training.collate import collate_fn
from src.evaluation.metrics import compute_subtoken_f1, compute_exact_match, normalize_name


def to_dev(batch, device):
    out = {}
    for k in ('block_tokens', 'edge_index', 'ext_call_ids', 'block_features', 'callee_tokens',
              'caller_tokens', 'string_tokens', 'binary_ext_ids', 'decoder_target'):
        if k in batch and torch.is_tensor(batch[k]):
            out[k] = batch[k].to(device)
    return out


@torch.no_grad()
def embed_and_decode(model, loader, device, sp, beam=5, use_amp=False, decode=True):
    model.eval()
    Z, preds, scores = [], [], []
    sos, eos = sp.bos_id(), sp.eos_id()
    for batch in loader:
        b = to_dev(batch, device)
        with torch.amp.autocast('cuda', enabled=use_amp):
            be = model.block_encoder(b['block_tokens'], block_features=b.get('block_features'))
            B, N, D = be.shape
            x = be.view(B * N, D)
            batch_vec = torch.arange(B, device=x.device).repeat_interleave(N)
            f, _, _ = model.graph_encoder(x, b['edge_index'], batch_vec)
            z = f
            if model.ext_encoder_enabled and model.fusion is not None:
                c = model.ext_encoder(b['ext_call_ids'])
                z, _ = model.fusion(z, c, has_ext_calls=model._compute_has_ext_calls(b['ext_call_ids']))
            if model.callee_encoder_enabled and 'callee_tokens' in b:
                ctx, has = model.callee_encoder(b['callee_tokens'])
                g = model.callee_gate(torch.cat([z, ctx], dim=1)); zf = g * z + (1 - g) * ctx
                m = has.unsqueeze(1).float(); z = m * zf + (1 - m) * z
            if model.caller_encoder_enabled and 'caller_tokens' in b:
                ctx, has = model.caller_encoder(b['caller_tokens'])
                g = model.caller_gate(torch.cat([z, ctx], dim=1)); zf = g * z + (1 - g) * ctx
                m = has.unsqueeze(1).float(); z = m * zf + (1 - m) * z
            if getattr(model, 'string_encoder_enabled', False) and 'string_tokens' in b:
                ctx, has = model.string_encoder(b['string_tokens'])
                g = model.string_gate(torch.cat([z, ctx], dim=1)); zf = g * z + (1 - g) * ctx
                m = has.unsqueeze(1).float(); z = m * zf + (1 - m) * z
        z = z.float()
        Z.append(z.cpu())
        if decode:
            for i in range(B):
                toks, sc = model.decoder.generate(z[i:i + 1], sos, eos, beam)
                preds.append(sp.decode(toks) if toks else '')
                scores.append(float(sc))
    return torch.cat(Z).numpy(), preds, scores


def knn(query, index, k=20, chunk=4096, device='cpu'):
    """cosine top-k; returns (sims [n,k], idx [n,k])."""
    idx_t = torch.from_numpy(index).to(device).half()
    idx_t = torch.nn.functional.normalize(idx_t.float(), dim=1).half()
    S, I = [], []
    for s in range(0, len(query), chunk):
        q = torch.from_numpy(query[s:s + chunk]).to(device).float()
        q = torch.nn.functional.normalize(q, dim=1).half()
        sim = q @ idx_t.T
        v, ix = sim.float().topk(min(k, idx_t.shape[0]), dim=1)
        S.append(v.cpu()); I.append(ix.cpu())
    return torch.cat(S).numpy(), torch.cat(I).numpy()


def f1_of(pred, gold, aliases=()):
    best = compute_subtoken_f1(pred, gold)
    for a in aliases:
        best = max(best, compute_subtoken_f1(pred, a))
    return best


def em_of(pred, gold, aliases=()):
    return compute_exact_match(pred, gold) or any(compute_exact_match(pred, a) for a in aliases)


def aurc(conf, risk):
    """area under risk-coverage curve (lower is better); risk = 1 - F1 per function."""
    order = np.argsort(-np.asarray(conf))
    r = np.asarray(risk)[order]
    cum = np.cumsum(r) / (np.arange(len(r)) + 1)
    return float(np.mean(cum))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--tiers', nargs='+', default=['val_xproj', 'xproject', 'test'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--k', type=int, default=20)
    ap.add_argument('--beam', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--tau', type=float, default=None, help='gate threshold; tuned on val_xproj if omitted')
    ap.add_argument('--index-cache', default=None, help='npz path to cache train embeddings')
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = yaml.safe_load(open(args.config))
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    ck_cfg = ck.get('config', cfg)
    sp = VotesTokenizer(vocab_path=cfg['data']['votes_vocab_path'])
    split_file = cfg['data'].get('split_file', 'data/split_v2.json')
    split = json.load(open(split_file))
    ds = FunctionDatasetV2(match_index_path=cfg['data'].get('match_index_path', 'data/match_index_v2.json'),
                           string_refs_dir=cfg['data'].get('string_refs_dir', 'data/string_refs_v2'),
                           votes_vocab_path=cfg['data']['votes_vocab_path'],
                           token_vocab=ck['token_vocab'], ext_vocab=ck['ext_vocab'],
                           max_blocks=cfg['data']['max_blocks_per_function'], max_tokens=cfg['data']['max_tokens_per_block'],
                           max_name_len=cfg['data']['max_name_length'], min_tokens=cfg['data'].get('min_tokens', 1),
                           corpora=set(cfg['data']['corpora']) if cfg['data'].get('corpora') else None,
                           vocab_binaries=set(split['train']))
    if ck.get('split_sha256') and ck['split_sha256'] != __import__('hashlib').sha256(open(split_file, 'rb').read()).hexdigest():
        print('WARNING: checkpoint was trained with a different split file (sha mismatch)')
    train_idx, val_idx, test_idx = ds.get_splits(split_file=split_file)
    tier_idx = {'train': train_idx, 'val_indist': val_idx, 'test': test_idx, 'val_xproj': list(ds.val_xproj_idx)}
    xp_bins = set(split.get('xproject', []))
    tier_idx['xproject'] = [i for i, s in enumerate(ds.samples) if s['binary'] in xp_bins]
    ds.training_mode = False

    model = FunctionNamer(ck_cfg).to(device)
    model.load_state_dict(ck['model_state_dict']); model.eval()
    print(f"model {sum(p.numel() for p in model.parameters()):,} params; tiers: " +
          ', '.join(f'{t}={len(tier_idx[t])}' for t in ['train'] + args.tiers))

    def loader(idx):
        return DataLoader(Subset(ds, idx), batch_size=args.batch_size, shuffle=False,
                          collate_fn=collate_fn, num_workers=args.num_workers)
    # ---- train index
    t0 = time.time()
    if args.index_cache and os.path.exists(args.index_cache):
        d = np.load(args.index_cache, allow_pickle=True); Ztr = d['z']; tr_names = list(d['names']); tr_bins = list(d['bins'])
    else:
        Ztr, _, _ = embed_and_decode(model, loader(train_idx), device, sp, decode=False, use_amp=args.amp)
        tr_names = [ds.samples[i]['name'] for i in train_idx]; tr_bins = [ds.samples[i]['binary'] for i in train_idx]
        if args.index_cache:
            np.savez(args.index_cache, z=Ztr, names=np.array(tr_names, dtype=object), bins=np.array(tr_bins, dtype=object))
    print(f'train index {Ztr.shape} in {time.time() - t0:.0f}s')
    train_names = set(tr_names)
    train_subtoks = set()
    for n in train_names:
        train_subtoks.update(t for t in split_name_to_subtokens(n) if t != '_')
    regime = split.get('meta', {}).get('roles', {})

    # ---- per tier
    records = {}
    for tier in args.tiers:
        idx = tier_idx[tier]
        if not idx:
            print(f'{tier}: empty'); continue
        t0 = time.time()
        Z, dec, dsc = embed_and_decode(model, loader(idx), device, sp, beam=args.beam, use_amp=args.amp)
        sims, nn = knn(Z, Ztr, k=args.k, device=device)
        rows = []
        for j, i in enumerate(idx):
            s = ds.samples[i]
            gold = s['name']; al = s.get('aliases', [])
            top = [tr_names[t] for t in nn[j]]
            knn1 = top[0]
            # neighbour vote among top-5 (name agreement)
            vote = max(set(top[:5]), key=lambda n: (top[:5].count(n), -top[:5].index(n)))
            subs = [t for t in split_name_to_subtokens(gold) if t != '_']
            seen = gold in train_names or any(a in train_names for a in al)
            stratum = 'seen_name' if seen else ('novel_composable' if subs and all(t in train_subtoks for t in subs) else 'oov')
            rows.append({'binary': s['binary'], 'package': s['binary'].split('_')[0], 'name': gold, 'aliases': al,
                         'in_dynsym': bool(s.get('in_dynsym')), 'binding': s.get('binding'), 'stratum': stratum,
                         'regime': regime.get(s['binary'].split('_')[0], tier),
                         'dec': dec[j], 'dec_score': dsc[j], 'knn1': knn1, 'knn1_sim': float(sims[j][0]),
                         'knn_margin': float(sims[j][0] - sims[j][1]) if sims.shape[1] > 1 else 0.0,
                         'knn5_vote': vote, 'knn5_agree': top[:5].count(knn1) / 5.0,
                         'knn_top': top[:10], 'knn_top_sims': [float(x) for x in sims[j][:10]],
                         'f1_dec': f1_of(dec[j], gold, al), 'f1_knn': f1_of(knn1, gold, al),
                         'em_dec': em_of(dec[j], gold, al), 'em_knn': em_of(knn1, gold, al),
                         'n_tokens': sum(b['num_tokens'] for b in s['graph']['blocks'])})
        records[tier] = rows
        print(f'{tier}: {len(rows)} fns in {time.time() - t0:.0f}s')

    # ---- gate threshold (on val_xproj if present else first tier), target set = not in_dynsym
    def target(rows):
        return [r for r in rows if not r['in_dynsym']]
    tune_rows = target(records.get('val_xproj') or records[args.tiers[0]])
    if args.tau is None:
        best = (-1, 0.5)
        for tau in np.arange(0.30, 1.0, 0.02):
            f = np.mean([r['f1_knn'] if r['knn1_sim'] >= tau else r['f1_dec'] for r in tune_rows])
            if f > best[0]:
                best = (f, float(tau))
        tau = best[1]
    else:
        tau = args.tau
    print(f'gate tau = {tau:.2f}')

    def summarize(rows):
        if not rows:
            return {}
        out = {'n': len(rows)}
        for h in ('dec', 'knn'):
            out[f'f1_{h}'] = float(np.mean([r[f'f1_{h}'] for r in rows]))
            out[f'em_{h}'] = float(np.mean([r[f'em_{h}'] for r in rows]))
        out['f1_oracle'] = float(np.mean([max(r['f1_dec'], r['f1_knn']) for r in rows]))
        out['f1_gate'] = float(np.mean([r['f1_knn'] if r['knn1_sim'] >= tau else r['f1_dec'] for r in rows]))
        out['em_gate'] = float(np.mean([r['em_knn'] if r['knn1_sim'] >= tau else r['em_dec'] for r in rows]))
        out['gate_knn_share'] = float(np.mean([r['knn1_sim'] >= tau for r in rows]))
        pk = defaultdict(list)
        for r in rows:
            pk[r['package']].append(r)
        out['macro_f1_gate'] = float(np.mean([np.mean([r['f1_knn'] if r['knn1_sim'] >= tau else r['f1_dec'] for r in v]) for v in pk.values()]))
        out['macro_f1_dec'] = float(np.mean([np.mean([r['f1_dec'] for r in v]) for v in pk.values()]))
        out['macro_f1_knn'] = float(np.mean([np.mean([r['f1_knn'] for r in v]) for v in pk.values()]))
        out['aurc_knn'] = aurc([r['knn1_sim'] for r in rows], [1 - r['f1_knn'] for r in rows])
        out['aurc_dec'] = aurc([r['dec_score'] for r in rows], [1 - r['f1_dec'] for r in rows])
        return out

    summary = {'ckpt': args.ckpt, 'tau': tau, 'k': args.k, 'tiers': {}}
    for tier, rows in records.items():
        tgt = target(rows)
        S = {'all_incl_dynsym': summarize(rows), 'target': summarize(tgt),
             'symbol_visible': summarize([r for r in rows if r['in_dynsym']]),
             'by_stratum': {st: summarize([r for r in tgt if r['stratum'] == st]) for st in ('seen_name', 'novel_composable', 'oov')},
             'by_regime': {}, 'by_package': {}}
        for key in ('regime', 'package'):
            groups = defaultdict(list)
            for r in tgt:
                groups[r[key]].append(r)
            S['by_' + key] = {g: summarize(v) for g, v in sorted(groups.items())}
        summary['tiers'][tier] = S
        t = S['target']
        print(f"[{tier}] target n={t.get('n')}  F1 dec {t.get('f1_dec', 0):.4f} | knn {t.get('f1_knn', 0):.4f} | gate {t.get('f1_gate', 0):.4f} "
              f"| oracle {t.get('f1_oracle', 0):.4f}  EM gate {t.get('em_gate', 0):.4f}  macroF1 gate {t.get('macro_f1_gate', 0):.4f}  "
              f"AURC knn {t.get('aurc_knn', 0):.3f} dec {t.get('aurc_dec', 0):.3f}")
        for st, v in S['by_stratum'].items():
            if v:
                print(f"    {st:17s} n={v['n']:6d} dec {v['f1_dec']:.4f} knn {v['f1_knn']:.4f} gate {v['f1_gate']:.4f}")
        for g, v in S['by_regime'].items():
            print(f"    regime {g:14s} n={v['n']:6d} dec {v['f1_dec']:.4f} knn {v['f1_knn']:.4f} gate {v['f1_gate']:.4f} macro {v['macro_f1_gate']:.4f}")
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    json.dump(summary, open(args.out, 'w'), indent=1)
    with open(args.out.replace('.json', '') + '_records.jsonl', 'w') as fh:
        for tier, rows in records.items():
            for r in rows:
                r2 = dict(r, tier=tier); fh.write(json.dumps(r2) + '\n')
    print('wrote', args.out)


if __name__ == '__main__':
    main()
