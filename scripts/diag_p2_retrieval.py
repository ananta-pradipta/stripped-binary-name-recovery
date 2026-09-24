#!/usr/bin/env python3
"""P2 diagnosis: will the Dual-Head phase work? (run BEFORE any P3 training)

Measures, on the frozen P2 encoder (no retraining):
  1. RETRIEVAL head: k-NN over train fused embeddings z -> top-1 name; F1/EM per
     regime (FT/NCT) and per name-stratum (seen/novel), vs the decoder's dump.
  2. HYBRID ORACLE: max(decoder, retrieval) per function = ceiling of a perfect router.
  3. NAME-VOCAB ORACLE (sampled): best achievable F1 copying ANY train name = ceiling
     of any retrieval-flavoured system.
  4. ROUTER separability: AUC of simple features (retrieval margin, top1 sim) for
     (a) retrieval-beats-decoder, (b) name-is-novel; risk-coverage curve for abstention.

Usage: python3 scripts/diag_p2_retrieval.py CKPT --decoder-preds results/p2_preds_greedy.tsv \
          [--config ...] [--sample-oracle 4000] [--save results/diag_p2.json]
"""
import argparse, json, os, sys, random
from collections import defaultdict, Counter
import torch, yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.function_namer import FunctionNamer
from src.preprocessing.dataset_v2 import FunctionDatasetV2
from src.training.collate import collate_fn
from src.evaluation.metrics import compute_subtoken_f1, split_name


def _to(batch, key, device):
    v = batch.get(key)
    return v.to(device) if v is not None else None


def extract_embeddings(model, dataset, idx, device, batch_size=256, use_amp=True, num_workers=4):
    model.eval()
    out = []
    loader = DataLoader(Subset(dataset, idx), batch_size=batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=num_workers)
    with torch.no_grad():
        for batch in tqdm(loader, desc='embed', leave=False):
            with torch.amp.autocast('cuda', enabled=use_amp):
                _, _, _, z, _ = model(batch['block_tokens'].to(device), batch['edge_index'].to(device),
                                      batch['ext_call_ids'].to(device), batch['decoder_input'].to(device),
                                      teacher_forcing_ratio=1.0,
                                      block_features=_to(batch, 'block_features', device),
                                      callee_tokens=_to(batch, 'callee_tokens', device),
                                      caller_tokens=_to(batch, 'caller_tokens', device),
                                      string_tokens=_to(batch, 'string_tokens', device),
                                      binary_ext_ids=_to(batch, 'binary_ext_ids', device))
            out.append(torch.nn.functional.normalize(z.float(), dim=1).cpu())
    return torch.cat(out)


def knn(train_emb, query_emb, device, k=5, chunk=8192):
    """Return (top-k sims, top-k train indices) for each query."""
    tr = train_emb.to(device)
    sims, idxs = [], []
    for i in range(0, len(query_emb), chunk):
        q = query_emb[i:i + chunk].to(device)
        s = q @ tr.T
        v, ix = s.topk(k, dim=1)
        sims.append(v.cpu()); idxs.append(ix.cpu())
    return torch.cat(sims), torch.cat(idxs)


def auc(pos, neg):
    """Mann-Whitney AUC of score separating pos (should be higher) from neg."""
    import bisect
    neg_s = sorted(neg)
    n = len(pos) * len(neg)
    if n == 0:
        return float('nan')
    wins = sum(bisect.bisect_left(neg_s, p) + 0.5 * (bisect.bisect_right(neg_s, p) - bisect.bisect_left(neg_s, p))
               for p in pos)
    return wins / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint')
    ap.add_argument('--config', default=None)
    ap.add_argument('--decoder-preds', required=True)
    ap.add_argument('--sample-oracle', type=int, default=4000)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--save', default='results/diag_p2.json')
    ap.add_argument('--dump', default=None, help='TSV per-function retrieval dump')
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = random.Random(42)

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = yaml.safe_load(open(args.config)) if args.config else ckpt['config']
    split_file = cfg['data']['split_file']
    split = json.load(open(split_file))
    regime = dict(split['meta'].get('test_regime', {}))
    for p, d in split['meta']['tiers'].get('val_xproj', {}).items():
        regime.setdefault(p, d['regime'])

    dataset = FunctionDatasetV2(
        match_index_path=cfg['data']['match_index_path'], string_refs_dir=cfg['data'].get('string_refs_dir'),
        votes_vocab_path=cfg['data']['votes_vocab_path'],
        max_blocks=cfg['data']['max_blocks_per_function'], max_tokens=cfg['data']['max_tokens_per_block'],
        max_name_len=cfg['data']['max_name_length'], min_tokens=cfg['data'].get('min_tokens', 1),
        token_vocab=ckpt['token_vocab'], ext_vocab=ckpt['ext_vocab'], vocab_binaries=set(split['train']),
        cache_path=cfg['data'].get('cache_path'),
        train_pkg_cap=cfg['data'].get('train_pkg_cap'))
    train_idx, val_idx, test_idx = dataset.get_splits(cfg['data']['train_split'], cfg['data']['val_split'],
                                                      split_file=split_file)
    if not val_idx and dataset.val_xproj_idx:
        val_idx = list(dataset.val_xproj_idx)
    train_idx, val_idx, test_idx = dataset.apply_split_policy(train_idx, val_idx, test_idx)
    train_names = [dataset.samples[i]['name'] for i in train_idx]
    train_name_set = set(train_names)

    # decoder predictions (join on binary+bap_name)
    dec = {}
    with open(args.decoder_preds) as fh:
        next(fh)
        for line in fh:
            tier, binary, bap, true, pred, reg, seen, f1 = line.rstrip('\n').split('\t')
            dec[(binary, bap)] = (pred, float(f1))

    st = ckpt['model_state_dict']
    cfg['block_encoder']['token_vocab_size'] = len(dataset.token_vocab)
    cfg['graph_encoder']['input_dim'] = cfg['block_encoder']['output_dim']
    if 'ext_encoder.embedding.weight' in st:
        cfg['external_encoder']['vocab_size'] = st['ext_encoder.embedding.weight'].shape[0]
    cfg['decoder']['bpe_vocab_size'] = st['decoder.embedding.weight'].shape[0]
    model = FunctionNamer(cfg).to(device)
    model.load_state_dict(st)

    use_amp = device.type == 'cuda'
    tr_emb = extract_embeddings(model, dataset, train_idx, device, args.batch_size, use_amp, args.num_workers)
    report = {'checkpoint': args.checkpoint, 'n_train': len(train_idx), 'tiers': {}}
    dump_rows = []
    for tier, idx in (('val', val_idx), ('test', test_idx)):
        q_emb = extract_embeddings(model, dataset, idx, device, args.batch_size, use_amp, args.num_workers)
        sims, nbrs = knn(tr_emb, q_emb, device, k=5)
        rows = []
        for j, i in enumerate(tqdm(idx, desc=f'score {tier}', leave=False)):
            s = dataset.samples[i]
            true = s['name']
            pkg = s['binary'].split('_')[0]
            r_name = train_names[nbrs[j, 0]]
            r_f1 = compute_subtoken_f1(r_name, true)
            top5 = [train_names[nbrs[j, m]] for m in range(5)]
            r5_f1 = max(compute_subtoken_f1(n, true) for n in top5)
            d_pred, d_f1 = dec.get((s['binary'], s['bap_name']), ('', 0.0))
            rows.append({'pkg': pkg, 'regime': regime.get(pkg, '?'), 'seen': true in train_name_set,
                         'true': true, 'd_pred': d_pred, 'd_f1': d_f1, 'r_pred': r_name, 'r_f1': r_f1,
                         'r5_f1': r5_f1, 'sim1': float(sims[j, 0]), 'margin': float(sims[j, 0] - sims[j, 1])})
            if args.dump:
                dump_rows.append((tier, s['binary'], s['bap_name'], true, d_pred, f"{d_f1:.3f}",
                                  r_name, f"{r_f1:.3f}", f"{float(sims[j,0]):.4f}", f"{float(sims[j,0]-sims[j,1]):.4f}"))

        def agg(sub):
            n = len(sub)
            if not n:
                return {'n': 0}
            return {'n': n, 'decoder_f1': sum(x['d_f1'] for x in sub) / n,
                    'retrieval_f1': sum(x['r_f1'] for x in sub) / n,
                    'retrieval_top5_oracle_f1': sum(x['r5_f1'] for x in sub) / n,
                    'hybrid_oracle_f1': sum(max(x['d_f1'], x['r_f1']) for x in sub) / n,
                    'retrieval_em': sum(x['r_pred'] == x['true'] for x in sub) / n,
                    'decoder_em': sum(x['d_pred'] == x['true'] for x in sub) / n}
        t = {'overall': agg(rows)}
        for reg in ('FT', 'NCT'):
            t[reg] = agg([x for x in rows if x['regime'] == reg])
        for k, c in (('seen_name', True), ('novel_name', False)):
            t[k] = agg([x for x in rows if x['seen'] == c])
        # per-package macro
        g = defaultdict(list)
        for x in rows:
            g[x['pkg']].append(x)
        pk = {p: agg(v) for p, v in g.items()}
        t['macro'] = {m: sum(v[m] for v in pk.values()) / len(pk)
                      for m in ('decoder_f1', 'retrieval_f1', 'hybrid_oracle_f1')}
        t['per_pkg'] = {p: {m: round(v[m], 4) for m in ('n', 'decoder_f1', 'retrieval_f1', 'hybrid_oracle_f1')}
                        for p, v in sorted(pk.items())}
        # router separability
        rb = [x for x in rows if x['r_f1'] > x['d_f1']]
        db = [x for x in rows if x['r_f1'] < x['d_f1']]
        t['router'] = {
            'retrieval_beats_decoder_pct': len(rb) / len(rows),
            'decoder_beats_retrieval_pct': len(db) / len(rows),
            'auc_margin_retrieval_wins': auc([x['margin'] for x in rb], [x['margin'] for x in db]),
            'auc_sim1_seen_vs_novel': auc([x['sim1'] for x in rows if x['seen']],
                                          [x['sim1'] for x in rows if not x['seen']]),
            'auc_sim1_retrieval_correct': auc([x['sim1'] for x in rows if x['r_f1'] >= 0.5],
                                              [x['sim1'] for x in rows if x['r_f1'] < 0.5]),
        }
        # abstention risk-coverage on sim1: keep top q% by sim1, report retrieval F1 of kept
        by_sim = sorted(rows, key=lambda x: -x['sim1'])
        t['risk_coverage_sim1'] = {f'{q}%': round(sum(x['r_f1'] for x in by_sim[:max(1, int(len(rows) * q / 100))]) /
                                                  max(1, int(len(rows) * q / 100)), 4)
                                   for q in (5, 10, 20, 30, 50, 100)}
        # sampled name-vocab oracle
        uniq_names = list({n for n in train_names})
        name_toks = {n: Counter(split_name(n)) for n in uniq_names}
        samp = rng.sample(rows, min(args.sample_oracle, len(rows)))
        oracle = []
        for x in tqdm(samp, desc=f'oracle {tier}', leave=False):
            tt = Counter(split_name(x['true']))
            tn = sum(tt.values())
            best = 0.0
            for n, ct in name_toks.items():
                inter = sum((tt & ct).values())
                if inter == 0:
                    continue
                f = 2 * inter / (tn + sum(ct.values()))
                if f > best:
                    best = f
                    if best >= 0.999:
                        break
            oracle.append(best)
        t['vocab_oracle_f1_sampled'] = sum(oracle) / len(oracle)
        t['vocab_oracle_n'] = len(oracle)
        report['tiers'][tier] = t
        print(f"\n=== {tier} ===")
        for k in ('overall', 'FT', 'NCT', 'seen_name', 'novel_name'):
            d = t[k]
            if d['n']:
                print(f"  {k:<11} n={d['n']:>7} decoder {d['decoder_f1']:.4f} | retrieval {d['retrieval_f1']:.4f} "
                      f"| top5-oracle {d['retrieval_top5_oracle_f1']:.4f} | hybrid-oracle {d['hybrid_oracle_f1']:.4f}")
        print(f"  macro: decoder {t['macro']['decoder_f1']:.4f} retrieval {t['macro']['retrieval_f1']:.4f} "
              f"hybrid-oracle {t['macro']['hybrid_oracle_f1']:.4f}")
        print(f"  router: {json.dumps({k: round(v, 4) for k, v in t['router'].items()})}")
        print(f"  risk-coverage(sim1): {t['risk_coverage_sim1']}")
        print(f"  vocab-oracle (n={t['vocab_oracle_n']}): {t['vocab_oracle_f1_sampled']:.4f}")
    if args.save:
        os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
        json.dump(report, open(args.save, 'w'), indent=1)
    if args.dump:
        with open(args.dump, 'w') as fh:
            fh.write('tier\tbinary\tbap\ttrue\td_pred\td_f1\tr_pred\tr_f1\tsim1\tmargin\n')
            for r in dump_rows:
                fh.write('\t'.join(r) + '\n')
    print('EFFECT: diag done', {t: round(d['overall']['hybrid_oracle_f1'], 4) for t, d in report['tiers'].items()})


if __name__ == '__main__':
    main()
