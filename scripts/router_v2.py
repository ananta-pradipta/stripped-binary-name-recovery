#!/usr/bin/env python3
"""P3-DualHead router v2: rich per-function features + learned head-choice/abstention.

One pass on the frozen P2-Baseline checkpoint:
  features per eval function:
    sim1, margin            (k-NN over train fused embeddings)
    ext_jacc                (Jaccard of PLT-call sets: function vs top-1 neighbour)
    str_jacc                (Jaccard of string-ref token sets vs top-1 neighbour)
    d_conf                  (decoder confidence: mean max-softmax over generated steps)
    d_len, n_ext, n_str, n_blocks
  router (trained on VAL only, logistic, standardized):
    head-choice  P(retrieval >= decoder)
    confidence   P(chosen output F1 >= 0.5)  -> abstention / risk-coverage / ECE
  reports TEST: hybrid F1 (micro/macro/regime), AURC, coverage table, ECE(10-bin).

Usage: python3 scripts/router_v2.py CKPT --decoder-preds results/p2_preds_greedy.tsv
"""
import argparse, json, math, os, sys, random
from collections import defaultdict
import torch, yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.function_namer import FunctionNamer
from src.preprocessing.dataset_v2 import FunctionDatasetV2
from src.training.collate import collate_fn
from src.evaluation.metrics import compute_subtoken_f1


def _to(batch, key, device):
    v = batch.get(key)
    return v.to(device) if v is not None else None


def forward_pass(model, dataset, idx, device, sp, batch_size, num_workers, use_amp):
    """Return (normalized z [N,D], decoder confidence [N], generated length [N])."""
    eos_id = sp.eos_id()
    model.eval()
    zs, confs, lens = [], [], []
    loader = DataLoader(Subset(dataset, idx), batch_size=batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=num_workers)
    with torch.no_grad():
        for batch in tqdm(loader, desc='fwd', leave=False):
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits, _, _, z, _ = model(batch['block_tokens'].to(device), batch['edge_index'].to(device),
                                           batch['ext_call_ids'].to(device), batch['decoder_input'].to(device),
                                           teacher_forcing_ratio=0.0,
                                           block_features=_to(batch, 'block_features', device),
                                           callee_tokens=_to(batch, 'callee_tokens', device),
                                           caller_tokens=_to(batch, 'caller_tokens', device),
                                           string_tokens=_to(batch, 'string_tokens', device),
                                           binary_ext_ids=_to(batch, 'binary_ext_ids', device))
            zs.append(torch.nn.functional.normalize(z.float(), dim=1).cpu())
            prob = torch.softmax(logits.float(), dim=-1)
            top_p, top_i = prob.max(-1)                      # [B, T]
            for b in range(top_i.shape[0]):
                seq_p, n = [], 0
                for t in range(top_i.shape[1]):
                    tid = int(top_i[b, t])
                    seq_p.append(float(top_p[b, t])); n += 1
                    if tid == eos_id:
                        break
                confs.append(sum(seq_p) / max(1, len(seq_p)))
                lens.append(n)
    return torch.cat(zs), confs, lens


def knn(train_emb, query_emb, device, k=2, chunk=8192):
    tr = train_emb.to(device)
    sims, idxs = [], []
    for i in range(0, len(query_emb), chunk):
        q = query_emb[i:i + chunk].to(device)
        v, ix = (q @ tr.T).topk(k, dim=1)
        sims.append(v.cpu()); idxs.append(ix.cpu())
    return torch.cat(sims), torch.cat(idxs)


def jacc(a, b):
    if not a and not b:
        return 0.0
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))


class Logistic:
    def __init__(self, feats):
        self.feats = feats
    def fit(self, data, labels, epochs=300, lr=0.5):
        F = self.feats
        self.mu = {f: sum(x[f] for x in data) / len(data) for f in F}
        self.sd = {f: (sum((x[f] - self.mu[f]) ** 2 for x in data) / len(data)) ** 0.5 or 1 for f in F}
        X = [[(x[f] - self.mu[f]) / self.sd[f] for f in F] for x in data]
        w = [0.0] * len(F); b = 0.0; n = len(data)
        for _ in range(epochs):
            gw = [0.0] * len(F); gb = 0.0
            for xi, yi in zip(X, labels):
                z = b + sum(wj * xj for wj, xj in zip(w, xi))
                p = 1 / (1 + math.exp(-max(-30, min(30, z))))
                e = p - yi; gb += e
                for k in range(len(F)):
                    gw[k] += e * xi[k]
            b -= lr * gb / n
            for k in range(len(F)):
                w[k] -= lr * gw[k] / n
        self.w, self.b = w, b
        return self
    def __call__(self, x):
        z = self.b + sum(wj * (x[f] - self.mu[f]) / self.sd[f] for wj, f in zip(self.w, self.feats))
        return 1 / (1 + math.exp(-max(-30, min(30, z))))


def aurc(scores, outcomes):
    n = len(scores)
    o = sorted(range(n), key=lambda i: -scores[i])
    cum = 0.0; area = 0.0
    for k, i in enumerate(o, 1):
        cum += 1 - outcomes[i]
        area += cum / k
    return area / n


def ece(confs, corrects, bins=10):
    tot = len(confs); e = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [i for i, c in enumerate(confs) if lo <= c < hi or (b == bins - 1 and c == 1.0)]
        if not sel:
            continue
        acc = sum(corrects[i] for i in sel) / len(sel)
        conf = sum(confs[i] for i in sel) / len(sel)
        e += len(sel) / tot * abs(acc - conf)
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint')
    ap.add_argument('--config', default=None)
    ap.add_argument('--decoder-preds', required=True)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--save', default='results/router_v2.json')
    ap.add_argument('--dump', default='results/router_v2_features.tsv')
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = yaml.safe_load(open(args.config)) if args.config else ckpt['config']
    split = json.load(open(cfg['data']['split_file']))
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
        enrich_a3=bool(cfg['data'].get('enrich_a3', False)),
        rodata_consts_dir=cfg['data'].get('rodata_consts_dir'))
    sp = dataset.sp
    train_idx, val_idx, test_idx = dataset.get_splits(cfg['data']['train_split'], cfg['data']['val_split'],
                                                      split_file=cfg['data']['split_file'])
    if not val_idx and dataset.val_xproj_idx:
        val_idx = list(dataset.val_xproj_idx)
    train_idx, val_idx, test_idx = dataset.apply_split_policy(train_idx, val_idx, test_idx)

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

    tr_emb, _, _ = forward_pass(model, dataset, train_idx, device, sp, args.batch_size, args.num_workers, use_amp)
    tr_meta = [(dataset.samples[i]['name'], dataset.samples[i]['ext_calls'],
                dataset.string_refs.get((dataset.samples[i]['binary'], dataset.samples[i]['bap_name']), []))
               for i in train_idx]

    def build_rows(idx, tier):
        q_emb, confs, lens = forward_pass(model, dataset, idx, device, sp, args.batch_size, args.num_workers, use_amp)
        sims, nbrs = knn(tr_emb, q_emb, device, k=2)
        rows = []
        for j, i in enumerate(idx):
            s = dataset.samples[i]
            nb = int(nbrs[j, 0])
            r_name, nb_ext, nb_str = tr_meta[nb]
            my_str = dataset.string_refs.get((s['binary'], s['bap_name']), [])
            d_pred, d_f1 = dec.get((s['binary'], s['bap_name']), ('', 0.0))
            rows.append({
                'tier': tier, 'binary': s['binary'], 'bap': s['bap_name'], 'pkg': s['binary'].split('_')[0],
                'regime': regime.get(s['binary'].split('_')[0], '?'), 'true': s['name'],
                'r_pred': r_name, 'r_f1': compute_subtoken_f1(r_name, s['name']),
                'd_pred': d_pred, 'd_f1': d_f1,
                'sim1': float(sims[j, 0]), 'margin': float(sims[j, 0] - sims[j, 1]),
                'ext_jacc': jacc(s['ext_calls'], nb_ext), 'str_jacc': jacc(my_str, nb_str),
                'd_conf': confs[j], 'd_len': lens[j],
                'n_ext': len(s['ext_calls']), 'n_str': len(my_str), 'n_blocks': len(s['graph']['blocks']),
            })
        return rows

    val_rows = build_rows(val_idx, 'val')
    test_rows = build_rows(test_idx, 'test')

    FEATS = ['sim1', 'margin', 'ext_jacc', 'str_jacc', 'd_conf', 'd_len', 'n_ext', 'n_str', 'n_blocks']
    head = Logistic(FEATS).fit(val_rows, [1.0 if x['r_f1'] >= x['d_f1'] else 0.0 for x in val_rows])
    # confidence model must be fit on the head-choice OUTPUT of the router itself
    def chosen_f1(x):
        return x['r_f1'] if head(x) >= 0.5 else x['d_f1']
    conf = Logistic(FEATS).fit(val_rows, [1.0 if chosen_f1(x) >= 0.5 else 0.0 for x in val_rows])

    report = {'checkpoint': args.checkpoint, 'features': FEATS,
              'head_weights': dict(zip(FEATS, head.w)) | {'bias': head.b}, 'tiers': {}}
    for tier, rows in (('val', val_rows), ('test', test_rows)):
        n = len(rows)
        r = sum(x['r_f1'] for x in rows) / n
        d = sum(x['d_f1'] for x in rows) / n
        hyb = [chosen_f1(x) for x in rows]
        pick_d = [x for x in rows if head(x) < 0.5]
        h = sum(hyb) / n
        cf = [conf(x) for x in rows]
        cor = [1.0 if f >= 0.5 else 0.0 for f in hyb]
        g = defaultdict(list)
        for x, f in zip(rows, hyb):
            g[x['pkg']].append(f)
        t = {'n': n, 'retrieval_f1': r, 'decoder_f1': d, 'hybrid_f1': h,
             'delta_vs_best_single': h - max(r, d),
             'router_picks_decoder_pct': len(pick_d) / n,
             'decoder_picks_correct': (sum(1 for x in pick_d if x['d_f1'] > x['r_f1']) / max(1, len(pick_d))),
             'macro_hybrid_f1': sum(sum(v) / len(v) for v in g.values()) / len(g),
             'aurc_conf_router': aurc(cf, hyb), 'aurc_sim1_retrieval': aurc([x['sim1'] for x in rows],
                                                                           [x['r_f1'] for x in rows]),
             'ece_conf': ece(cf, cor),
             'coverage': {}}
        order = sorted(range(n), key=lambda i: -cf[i])
        for q in (5, 10, 20, 30, 50, 100):
            k = max(1, n * q // 100)
            t['coverage'][f'{q}%'] = round(sum(hyb[i] for i in order[:k]) / k, 4)
        for regn in ('FT', 'NCT'):
            sub = [f for x, f in zip(rows, hyb) if x['regime'] == regn]
            if sub:
                t[f'hybrid_{regn}'] = sum(sub) / len(sub)
        report['tiers'][tier] = t
        print(f"\n=== {tier} === n={n}")
        for k, v in t.items():
            if k != 'coverage':
                print(f"  {k}: {v if isinstance(v, int) else round(v, 4) if isinstance(v, float) else v}")
        print(f"  coverage: {t['coverage']}")
    os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
    json.dump(report, open(args.save, 'w'), indent=1)
    if args.dump:
        with open(args.dump, 'w') as fh:
            cols = ['tier', 'binary', 'bap', 'true', 'r_pred', 'r_f1', 'd_pred', 'd_f1'] + FEATS
            fh.write('\t'.join(cols) + '\n')
            for x in val_rows + test_rows:
                fh.write('\t'.join(str(round(x[c], 4)) if isinstance(x[c], float) else str(x[c]) for c in cols) + '\n')
    print('EFFECT: router_v2 done',
          {t: (round(d['hybrid_f1'], 4), round(d['delta_vs_best_single'], 4)) for t, d in report['tiers'].items()})


if __name__ == '__main__':
    main()
