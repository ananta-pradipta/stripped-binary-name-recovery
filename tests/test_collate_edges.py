"""B1 regression test: batched edge_index must index into the padded node layout.

The model flattens block embeddings as (B * max_blocks) rows and builds
batch_vec = arange(B).repeat_interleave(max_blocks). Therefore sample i's edges
must be offset by i * max_blocks, NOT by the running sum of real num_blocks.
"""
import sys, torch
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.training.train import collate_fn

MAX_BLOCKS = 30

def make_sample(num_blocks, edges, T=20):
    return {
        'block_tokens': torch.zeros(MAX_BLOCKS, T, dtype=torch.long),
        'edge_index': torch.tensor(edges, dtype=torch.long).t().contiguous(),
        'num_blocks': num_blocks,
        'decoder_input': torch.tensor([1, 2, 3]),
        'decoder_target': torch.tensor([2, 3, 4]),
        'ext_call_ids': torch.tensor([0]),
    }

def test_edges_stay_inside_own_sample():
    batch = [make_sample(3, [(0, 1), (1, 2)]),
             make_sample(5, [(0, 1), (1, 4), (4, 2)]),
             make_sample(2, [(0, 1)])]
    out = collate_fn(batch)
    ei = out['edge_index']
    n0 = batch[0]['edge_index'].size(1)
    n1 = batch[1]['edge_index'].size(1)
    ranges = [(0, n0), (n0, n0 + n1), (n0 + n1, ei.size(1))]
    for i, (a, b) in enumerate(ranges):
        seg = ei[:, a:b]
        lo, hi = i * MAX_BLOCKS, (i + 1) * MAX_BLOCKS
        assert bool(((seg >= lo) & (seg < hi)).all()), (
            f"sample {i}: edges {seg.tolist()} not within node rows [{lo},{hi}) "
            f"-> edges point into another sample's blocks (B1 collate misalignment)")

if __name__ == '__main__':
    try:
        test_edges_stay_inside_own_sample(); print("PASS")
    except AssertionError as e:
        print("FAIL:", e); sys.exit(1)
