#!/usr/bin/env python3
"""Dump fused embeddings + top-K train neighbours (names, sims, string-Jaccard) for
zero-training A1 experiments (string rerank / candidate emission). Reuses diag machinery."""
import argparse, json, os, sys
import numpy as np, torch, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.diag_p2_retrieval import extract_embeddings, knn
from src.preprocessing.dataset_v2 import FunctionDatasetV2
from src.models.function_namer import FunctionNamer

ap = argparse.ArgumentParser()
ap.add_argument('checkpoint')
ap.add_argument('--out', default='results/emb_v2')
ap.add_argument('--topk', type=int, default=20)
ap.add_argument('--batch-size', type=int, default=256)
ap.add_argument('--num-workers', type=int, default=4)
args = ap.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
cfg = ckpt['config']
split = json.load(open(cfg['data']['split_file']))
dataset = FunctionDatasetV2(
    match_index_path=cfg['data']['match_index_path'], string_refs_dir=cfg['data'].get('string_refs_dir'),
    votes_vocab_path=cfg['data']['votes_vocab_path'],
    max_blocks=cfg['data']['max_blocks_per_function'], max_tokens=cfg['data']['max_tokens_per_block'],
    max_name_len=cfg['data']['max_name_length'], min_tokens=cfg['data'].get('min_tokens', 1),
    token_vocab=ckpt['token_vocab'], ext_vocab=ckpt['ext_vocab'], vocab_binaries=set(split['train']),
    cache_path=cfg['data'].get('cache_path'),
    enrich_a3=bool(cfg['data'].get('enrich_a3', False)),
    rodata_consts_dir=cfg['data'].get('rodata_consts_dir'),
    train_pkg_cap=cfg['data'].get('train_pkg_cap'))
tr, va, te = dataset.get_splits(cfg['data']['train_split'], cfg['data']['val_split'], split_file=cfg['data']['split_file'])
if not va and dataset.val_xproj_idx:
    va = list(dataset.val_xproj_idx)
tr, va, te = dataset.apply_split_policy(tr, va, te)
st = ckpt['model_state_dict']
cfg['block_encoder']['token_vocab_size'] = len(dataset.token_vocab)
cfg['graph_encoder']['input_dim'] = cfg['block_encoder']['output_dim']
if 'ext_encoder.embedding.weight' in st:
    cfg['external_encoder']['vocab_size'] = st['ext_encoder.embedding.weight'].shape[0]
cfg['decoder']['bpe_vocab_size'] = st['decoder.embedding.weight'].shape[0]
model = FunctionNamer(cfg).to(device)
model.load_state_dict(st)
os.makedirs(args.out, exist_ok=True)
use_amp = device.type == 'cuda'
tr_emb = extract_embeddings(model, dataset, tr, device, args.batch_size, use_amp, args.num_workers)
np.save(f'{args.out}/train_emb.npy', tr_emb.numpy().astype('float16'))
def meta(i):
    s = dataset.samples[i]
    return {'binary': s['binary'], 'bap': s['bap_name'], 'name': s['name'],
            'strings': dataset.string_refs.get((s['binary'], s['bap_name']), []),
            'ext': s['ext_calls'][:20]}
json.dump([meta(i) for i in tr], open(f'{args.out}/train_meta.json', 'w'))
for tier, idx in (('val', va), ('test', te)):
    q = extract_embeddings(model, dataset, idx, device, args.batch_size, use_amp, args.num_workers)
    sims, nbrs = knn(tr_emb, q, device, k=args.topk)
    np.savez_compressed(f'{args.out}/{tier}_knn.npz', sims=sims.numpy().astype('float16'),
                        nbrs=nbrs.numpy().astype('int32'))
    json.dump([meta(i) for i in idx], open(f'{args.out}/{tier}_meta.json', 'w'))
    print(tier, len(idx), 'done')
print('EFFECT: emb dump done', args.out)
