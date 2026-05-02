#!/usr/bin/env python3
"""Inspect actual decoder outputs on lighttpd/tinycc/cvs to check for mode collapse."""
import sys, os, json, yaml
from collections import Counter

os.chdir('<shared-project-root>')
sys.path.insert(0, '<shared-project-root>')
sys.path.insert(0, '<shared-project-root>/scripts')

import torch
from src.models.function_namer import FunctionNamer
from src.preprocessing.build_dataset import FunctionDataset
from src.preprocessing.build_votes import VotesTokenizer
from eval_cross_project import (
    predict_binary_with_embeddings,
    load_functions_from_graphs, load_external_calls, load_ground_truth,
    resolve_thunks,
)

CKPT = '<shared-project-root>/checkpoints/best_model_pdec_xfl.pt'
CONFIG = '<shared-project-root>/configs/optimized_large.yaml'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

with open(CONFIG) as f:
    cfg = yaml.safe_load(f)

print(f'Loading checkpoint: {CKPT}')
ckpt = torch.load(CKPT, map_location=device, weights_only=False)
token_vocab = ckpt.get('token_vocab')
ext_vocab   = ckpt.get('ext_vocab')

votes_vocab_path = cfg['data'].get('votes_vocab_path', 'data/votes_vocab.json')
sp_model = VotesTokenizer(vocab_path=votes_vocab_path)

dataset = FunctionDataset(
    graphs_dir=cfg['data']['graphs_dir'],
    labels_dir=cfg['data']['labels_dir'],
    external_calls_dir=cfg['data']['external_calls_dir'],
    bpe_model_path=cfg['data']['bpe_model_path'],
    external_vocab_path=cfg['data']['external_vocab_path'],
    max_blocks=cfg['data']['max_blocks_per_function'],
    max_tokens=cfg['data']['max_tokens_per_block'],
    max_name_len=cfg['data']['max_name_length'],
    votes_vocab_path=votes_vocab_path,
    token_vocab=token_vocab,
    ext_vocab_override=ext_vocab,
)

ckpt_state = ckpt['model_state_dict']
cfg['block_encoder']['token_vocab_size'] = len(dataset.token_vocab)
cfg['graph_encoder']['input_dim'] = cfg['block_encoder']['output_dim']
if 'ext_encoder.embedding.weight' in ckpt_state:
    cfg['external_encoder']['vocab_size'] = ckpt_state['ext_encoder.embedding.weight'].shape[0]
cfg['decoder']['bpe_vocab_size'] = ckpt_state['decoder.embedding.weight'].shape[0]

encoder = FunctionNamer(cfg).to(device)
encoder.load_state_dict(ckpt['model_state_dict'])
encoder.eval()
print('Model loaded.')

demo_cfg = ckpt.get('config', cfg)
graphs_dirs = ['data/graphs', 'demo/graphs']
labels_dirs = ['data/labels', 'demo/labels']
ext_dirs    = ['data/external_calls', 'demo/external_calls']

targets = [
    ('lighttpd', 'lighttpd', 'O0'),
    ('tinycc', 'tcc', 'O0'),
    ('cvs', 'cvs', 'O0'),
]

for pkg, binary, opt in targets:
    bin_name = f'{pkg}_{binary}_{opt}'
    functions = load_functions_from_graphs(bin_name, graphs_dirs)
    if not functions:
        print(f'\n=== {bin_name}: NO GRAPHS ==='); continue
    ext_by_func = load_external_calls(bin_name, ext_dirs)
    gt = load_ground_truth(bin_name, labels_dirs)
    if not gt:
        print(f'\n=== {bin_name}: NO GT ==='); continue
    resolve_thunks(functions, ext_by_func)

    print(f'\n=== {bin_name} ({len(functions)} fns, {len(gt)} GT) ===')
    preds, _ = predict_binary_with_embeddings(
        functions, ext_by_func, encoder, token_vocab, ext_vocab,
        sp_model, demo_cfg, device, beam_width=5, use_amp=False
    )
    names = []
    shown = 0
    for pf in preds:
        addr = pf['address']
        if not addr.startswith('0x'):
            addr = '0x' + addr
        tn = gt.get(addr)
        if not tn:
            try:
                tn = gt.get(hex(int(addr, 16) - 4))
            except Exception:
                pass
        if not tn:
            bn = pf.get('bap_name', '')
            if bn and not bn.startswith('sub_') and not bn.startswith('.') and not bn.startswith('stub_'):
                tn = bn
        pn = pf['predicted_name']
        names.append(pn)
        if shown < 25:
            print(f'  T: {tn!r:<45}  O: {pn!r}')
            shown += 1
    c = Counter(names)
    print(f'  Total preds: {len(names)}  Unique: {len(c)}')
    print(f'  Top-10 predicted names:')
    for n, k in c.most_common(10):
        print(f'    {k:5d}  {n!r}')
