"""
Pre-training dataset for self-supervised learning on binary function graphs.

Two objectives:
  1. MLM: mask 15% of instruction-type tokens (80% [MASK], 10% random, 10% keep)
  2. Contrastive: pair same-name functions across optimization levels

Key design decisions:
  - LAZY loading: graph JSON files loaded in __getitem__, not __init__,
    to avoid OOM with 241K+ files
  - [MASK] token uses token_vocab_size as its ID (appended to existing vocab)
  - Contrastive pairs built from match_index.json grouped by real_name
"""
import json
import os
import random
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    """Dataset for self-supervised pre-training of block + graph encoders.

    Each item returns:
      - block_tokens: [N, T] with MLM masking applied
      - original_tokens: [N, T] before masking (MLM targets)
      - mask_positions: [N, T] bool mask of where masking was applied
      - edge_index: [2, E] CFG edges
      - num_blocks: int
      - pair_idx: index of positive pair sample (for contrastive), or -1

    Lazy loading: only graph file paths are stored in __init__,
    actual JSON is loaded per __getitem__ call.
    """

    def __init__(
        self,
        graphs_dir: str,
        match_index_path: str,
        token_vocab: Dict[str, int],
        max_blocks: int = 30,
        max_tokens: int = 20,
        mask_prob: float = 0.15,
        mask_token_ratio: float = 0.8,
        random_token_ratio: float = 0.1,
        pairs_path: Optional[str] = None,
    ):
        self.graphs_dir = graphs_dir
        self.max_blocks = max_blocks
        self.max_tokens = max_tokens
        self.mask_prob = mask_prob
        self.mask_token_ratio = mask_token_ratio
        self.random_token_ratio = random_token_ratio

        # Token vocabulary + MASK token
        self.token_vocab = token_vocab
        self.mask_token_id = max(token_vocab.values()) + 1
        self.vocab_size = self.mask_token_id + 1  # includes MASK
        self.pad_id = token_vocab.get('<PAD>', 0)
        self.unk_id = token_vocab.get('<UNK>', 1)

        # Real token IDs for random replacement (exclude PAD)
        self.real_token_ids = [v for k, v in token_vocab.items() if v != self.pad_id]

        # Load match index — maps graph_path -> {binary, real_name, ...}
        with open(match_index_path) as f:
            self.match_index = json.load(f)

        # Build sample list: (graph_path, real_name, binary)
        # Only include files that exist
        self.samples = []
        self.sample_to_idx = {}  # graph_path -> index in self.samples
        for graph_path, info in self.match_index.items():
            if os.path.exists(graph_path):
                idx = len(self.samples)
                self.samples.append({
                    'graph_path': graph_path,
                    'real_name': info['real_name'],
                    'binary': info['binary'],
                })
                self.sample_to_idx[graph_path] = idx

        print(f"PretrainDataset: {len(self.samples)} samples, "
              f"vocab_size={self.vocab_size} (MASK id={self.mask_token_id})")

        # Build contrastive pairs
        self.pairs = []  # list of (idx_a, idx_b)
        self._build_contrastive_pairs(pairs_path)

        # Build pair lookup: sample_idx -> list of partner indices
        self.pair_lookup = defaultdict(list)
        for a, b in self.pairs:
            self.pair_lookup[a].append(b)
            self.pair_lookup[b].append(a)

        print(f"Contrastive pairs: {len(self.pairs)}, "
              f"samples with pairs: {len(self.pair_lookup)}")

    def _build_contrastive_pairs(self, pairs_path: Optional[str]):
        """Build contrastive pairs from pre-computed file or match_index."""
        if pairs_path and os.path.exists(pairs_path):
            with open(pairs_path) as f:
                raw_pairs = json.load(f)
            # raw_pairs: list of [path_a, path_b]
            for pa, pb in raw_pairs:
                ia = self.sample_to_idx.get(pa)
                ib = self.sample_to_idx.get(pb)
                if ia is not None and ib is not None:
                    self.pairs.append((ia, ib))
            return

        # Build from match_index: group by (package, real_name)
        # where package is the part before the first '_' in binary name
        groups = defaultdict(list)
        for idx, sample in enumerate(self.samples):
            real_name = sample['real_name']
            binary = sample['binary']
            # Group by real_name across different binaries of the same package
            # Extract package: binary format is "package_program_OX"
            parts = binary.rsplit('_', 1)  # split off optimization level
            if len(parts) == 2 and parts[1] in ('O0', 'O1', 'O2', 'O3'):
                base = parts[0]  # package_program
            else:
                base = binary
            groups[(base, real_name)].append(idx)

        # Create all C(n,2) pairs within each group
        for key, indices in groups.items():
            if len(indices) > 1:
                for a, b in combinations(indices, 2):
                    self.pairs.append((a, b))

    def __len__(self):
        return len(self.samples)

    def _load_graph(self, graph_path: str) -> dict:
        """Lazy-load a single graph JSON file."""
        with open(graph_path) as f:
            return json.load(f)

    def _tokenize_blocks(self, graph: dict) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Convert graph blocks to token ID tensor.

        Returns:
            tokens: [N, T] long tensor of token IDs
            edge_index: [2, E] long tensor of edges
            num_blocks: int
        """
        blocks = graph['blocks'][:self.max_blocks]
        num_blocks = len(blocks)

        # Token IDs
        tokens = torch.zeros(self.max_blocks, self.max_tokens, dtype=torch.long)
        for i, block in enumerate(blocks):
            block_toks = block['tokens'][:self.max_tokens]
            for j, tok in enumerate(block_toks):
                tokens[i, j] = self.token_vocab.get(tok, self.unk_id)

        # Edges — filter to valid block indices
        edges = graph.get('edges', [])
        valid_edges = [(s, d) for s, d in edges
                       if s < num_blocks and d < num_blocks]
        if valid_edges:
            edge_index = torch.tensor(valid_edges, dtype=torch.long).t()
        else:
            # Self-loop on block 0 to avoid empty graph
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)

        return tokens, edge_index, num_blocks

    def _apply_mlm_mask(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply MLM masking to token tensor.

        Args:
            tokens: [N, T] original token IDs
        Returns:
            masked_tokens: [N, T] with masking applied
            original_tokens: [N, T] unchanged copy
            mask_positions: [N, T] bool, True at masked positions
        """
        original_tokens = tokens.clone()
        masked_tokens = tokens.clone()

        # Only mask non-PAD positions
        non_pad = (tokens != self.pad_id)

        # Random mask selection: 15% of non-PAD tokens
        rand_vals = torch.rand_like(tokens, dtype=torch.float)
        mask_positions = (rand_vals < self.mask_prob) & non_pad

        # Of masked positions: 80% -> [MASK], 10% -> random, 10% -> keep
        mask_type = torch.rand_like(tokens, dtype=torch.float)

        # [MASK] replacement (80%)
        replace_mask = mask_positions & (mask_type < self.mask_token_ratio)
        masked_tokens[replace_mask] = self.mask_token_id

        # Random replacement (10%)
        random_mask = mask_positions & (mask_type >= self.mask_token_ratio) & \
                      (mask_type < self.mask_token_ratio + self.random_token_ratio)
        num_random = random_mask.sum().item()
        if num_random > 0:
            random_ids = torch.tensor(
                [random.choice(self.real_token_ids) for _ in range(num_random)],
                dtype=torch.long
            )
            masked_tokens[random_mask] = random_ids

        # Keep original (10%) — already in masked_tokens from clone

        return masked_tokens, original_tokens, mask_positions

    def __getitem__(self, idx):
        sample = self.samples[idx]
        graph = self._load_graph(sample['graph_path'])

        tokens, edge_index, num_blocks = self._tokenize_blocks(graph)
        masked_tokens, original_tokens, mask_positions = self._apply_mlm_mask(tokens)

        # Find a contrastive pair partner (if any)
        partners = self.pair_lookup.get(idx, [])
        pair_idx = random.choice(partners) if partners else -1

        return {
            'block_tokens': masked_tokens,          # [N, T] with masking
            'original_tokens': original_tokens,      # [N, T] MLM targets
            'mask_positions': mask_positions,         # [N, T] bool
            'edge_index': edge_index,                 # [2, E]
            'num_blocks': num_blocks,
            'pair_idx': pair_idx,
            'sample_idx': idx,
        }

    def get_pair_sample(self, idx):
        """Load a sample specifically for use as a contrastive pair.

        Returns same structure as __getitem__ but WITHOUT MLM masking
        (we only need the graph embedding for contrastive loss).
        """
        sample = self.samples[idx]
        graph = self._load_graph(sample['graph_path'])
        tokens, edge_index, num_blocks = self._tokenize_blocks(graph)

        return {
            'block_tokens': tokens,       # [N, T] NO masking for pair
            'edge_index': edge_index,      # [2, E]
            'num_blocks': num_blocks,
        }


def pretrain_collate_fn(batch, dataset: 'PretrainDataset'):
    """Collate function for pre-training batches.

    Handles:
      - Stacking fixed-size block_tokens, original_tokens, mask_positions
      - Offsetting and concatenating variable-length edge_index
      - Loading and collating contrastive pair samples
    """
    B = len(batch)
    max_blocks = dataset.max_blocks
    max_tokens = dataset.max_tokens

    # Stack fixed-size tensors
    block_tokens = torch.stack([s['block_tokens'] for s in batch])        # [B, N, T]
    original_tokens = torch.stack([s['original_tokens'] for s in batch])  # [B, N, T]
    mask_positions = torch.stack([s['mask_positions'] for s in batch])    # [B, N, T]

    # Offset and concat edge_index
    edge_lists = []
    offset = 0
    for s in batch:
        ei = s['edge_index'].clone()
        ei = ei + offset
        edge_lists.append(ei)
        offset += max_blocks  # use max_blocks since tokens tensor is [N=max_blocks, T]
    edge_index = torch.cat(edge_lists, dim=1)  # [2, total_edges]

    # Batch vector for graph encoder
    batch_vec = torch.arange(B).repeat_interleave(max_blocks)  # [B*N]

    result = {
        'block_tokens': block_tokens,
        'original_tokens': original_tokens,
        'mask_positions': mask_positions,
        'edge_index': edge_index,
        'batch_vec': batch_vec,
        'num_blocks': [s['num_blocks'] for s in batch],
    }

    # Collect contrastive pairs
    pair_indices = [s['pair_idx'] for s in batch]
    has_pairs = [i for i, pi in enumerate(pair_indices) if pi >= 0]

    if has_pairs:
        # Load pair samples
        pair_samples = []
        anchor_indices = []
        for i in has_pairs:
            pair_data = dataset.get_pair_sample(pair_indices[i])
            pair_samples.append(pair_data)
            anchor_indices.append(i)

        P = len(pair_samples)
        pair_block_tokens = torch.stack([s['block_tokens'] for s in pair_samples])

        pair_edge_lists = []
        pair_offset = 0
        for s in pair_samples:
            ei = s['edge_index'].clone()
            ei = ei + pair_offset
            pair_edge_lists.append(ei)
            pair_offset += max_blocks
        pair_edge_index = torch.cat(pair_edge_lists, dim=1)

        pair_batch_vec = torch.arange(P).repeat_interleave(max_blocks)

        result['pair_block_tokens'] = pair_block_tokens       # [P, N, T]
        result['pair_edge_index'] = pair_edge_index            # [2, E']
        result['pair_batch_vec'] = pair_batch_vec              # [P*N]
        result['anchor_indices'] = anchor_indices              # list of int
    else:
        result['pair_block_tokens'] = None
        result['pair_edge_index'] = None
        result['pair_batch_vec'] = None
        result['anchor_indices'] = []

    return result
