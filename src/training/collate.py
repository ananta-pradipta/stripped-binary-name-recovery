"""Canonical batch collate for FunctionDataset samples.

Single source of truth — every training / evaluation / indexing script must
import ``collate_fn`` from here instead of carrying its own copy.

Node layout contract (see ``FunctionNamer.forward``): block embeddings are
flattened to ``B * max_blocks`` rows and ``batch_vec = arange(B).repeat_interleave(max_blocks)``.
Therefore sample *i*'s edges must be offset by ``i * max_blocks`` (the padded
stride), **not** by the running sum of real ``num_blocks``.  The old
``offset += sample['num_blocks']`` variant pointed every sample after the first
into an earlier sample's node rows (defect B1, found 2026-08-05, fixed 2026-08-17).
"""
import torch


def collate_fn(batch):
    """Custom collate for variable-length sequences (padded-stride edge offsets)."""
    keys = batch[0].keys()
    result = {}
    for k in keys:
        if isinstance(batch[0][k], torch.Tensor):
            if k == 'edge_index':
                stride = batch[0]['block_tokens'].size(0)  # == max_blocks (padded)
                edge_lists = []
                for i, sample in enumerate(batch):
                    assert sample['block_tokens'].size(0) == stride, \
                        "all samples in a batch must be padded to the same max_blocks"
                    ei = sample[k].clone()
                    if ei.numel():
                        ei = ei + i * stride
                    edge_lists.append(ei)
                result[k] = torch.cat(edge_lists, dim=1)
            elif k in ('decoder_input', 'decoder_target', 'ext_call_ids'):
                max_len = max(s[k].size(0) for s in batch)
                padded = torch.zeros(len(batch), max_len, dtype=batch[0][k].dtype)
                for i, s in enumerate(batch):
                    padded[i, :s[k].size(0)] = s[k]
                result[k] = padded
            else:
                result[k] = torch.stack([s[k] for s in batch])
        elif k == 'block_features':
            result[k] = torch.stack([s[k] for s in batch])
        elif k == 'num_blocks':
            result[k] = [s[k] for s in batch]
        else:
            result[k] = [s[k] for s in batch]
    return result
