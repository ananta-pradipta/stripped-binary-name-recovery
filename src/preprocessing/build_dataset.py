"""
Assemble the PyTorch dataset from preprocessed files.

Uses match_index.json (built by 03_preprocess.sh) as the central
data source — this already has correct address matching between
BAP graphs and ground truth labels.

Block Statistics Features (NEW):
  In addition to instruction-type tokens, each block now carries
  numerical statistics computed from the token sequence and graph
  structure. These capture information lost during tokenization
  (e.g., block size, call density) without reintroducing the
  32K raw vocabulary.

  Block-level (8 features from tokens):
    - num_tokens:        total instructions in block
    - unique_types:      number of distinct token types
    - call_density:      fraction of CALL_* tokens
    - mem_density:       fraction of MEM_READ + MEM_WRITE tokens
    - arith_density:     fraction of ARITH tokens
    - stack_density:     fraction of STACK_* tokens
    - branch_density:    fraction of BRANCH + COND_BRANCH tokens
    - has_return:        1 if block contains RETURN, else 0

  Inter-block (3 features from graph structure):
    - in_degree:         number of incoming CFG edges
    - out_degree:        number of outgoing CFG edges
    - relative_position: block index / total blocks (0.0 = entry, 1.0 = exit)
"""
import json
import os
import re
import random
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict, Counter

import torch
from torch.utils.data import Dataset
import sentencepiece as spm
from src.preprocessing.build_votes import VotesTokenizer


# Number of statistical features per block
NUM_BLOCK_FEATURES = 11


def compute_block_features(block_tokens: List[str], block_idx: int,
                           num_blocks: int, in_degree: int,
                           out_degree: int) -> List[float]:
    """Compute numerical features for a single basic block.

    Args:
        block_tokens: list of instruction-type token strings
        block_idx: position of this block in the function (0-indexed)
        num_blocks: total number of blocks in the function
        in_degree: number of incoming CFG edges to this block
        out_degree: number of outgoing CFG edges from this block

    Returns:
        List of NUM_BLOCK_FEATURES float values
    """
    n = len(block_tokens)
    if n == 0:
        return [0.0] * NUM_BLOCK_FEATURES

    counts = Counter(block_tokens)

    # Block-level features (from tokens)
    num_tokens = float(n)
    unique_types = float(len(counts))

    call_count = sum(v for k, v in counts.items() if k.startswith('CALL_'))
    mem_count = counts.get('MEM_READ', 0) + counts.get('MEM_WRITE', 0)
    arith_count = counts.get('ARITH', 0)
    stack_count = (counts.get('STACK_OP', 0) + counts.get('STACK_STORE', 0)
                   + counts.get('STACK_LOAD', 0) + counts.get('STACK_ACCESS', 0))
    branch_count = counts.get('BRANCH', 0) + counts.get('COND_BRANCH', 0)
    has_return = 1.0 if 'RETURN' in counts else 0.0

    call_density = call_count / n
    mem_density = mem_count / n
    arith_density = arith_count / n
    stack_density = stack_count / n
    branch_density = branch_count / n

    # Inter-block features (from graph structure)
    in_deg = float(in_degree)
    out_deg = float(out_degree)
    relative_pos = float(block_idx) / max(num_blocks - 1, 1)

    return [
        num_tokens,       # 0: block size (log-scaled later)
        unique_types,     # 1: token diversity
        call_density,     # 2: fraction of call instructions
        mem_density,      # 3: fraction of memory operations
        arith_density,    # 4: fraction of arithmetic
        stack_density,    # 5: fraction of stack operations
        branch_density,   # 6: fraction of branches
        has_return,       # 7: is this a return block?
        in_deg,           # 8: incoming edges
        out_deg,          # 9: outgoing edges
        relative_pos,     # 10: position in function (0=entry, 1=exit)
    ]


def compute_block_degrees(edges: List[List[int]], num_blocks: int):
    """Compute in-degree and out-degree for each block from edge list."""
    in_degrees = [0] * num_blocks
    out_degrees = [0] * num_blocks
    for src, dst in edges:
        if src < num_blocks:
            out_degrees[src] += 1
        if dst < num_blocks:
            in_degrees[dst] += 1
    return in_degrees, out_degrees


class FunctionDataset(Dataset):
    """Dataset of binary functions for name prediction."""

    def __init__(
        self,
        graphs_dir: str,
        labels_dir: str,
        external_calls_dir: str,
        bpe_model_path: str,
        external_vocab_path: str,
        max_blocks: int = 100,
        max_tokens: int = 64,
        max_name_len: int = 15,
        token_vocab: Optional[dict] = None,
        match_index_path: str = 'data/match_index.json',
        ext_vocab_override: Optional[dict] = None,
        votes_vocab_path: Optional[str] = None,
        enrich_callees: bool = False,
        callee_dropout: float = 0.1,
        string_refs_dir: Optional[str] = None,
        string_vocab_path: Optional[str] = None,
        match_index_override: Optional[dict] = None,
    ):
        self.max_blocks = max_blocks
        self.max_tokens = max_tokens
        self.max_name_len = max_name_len
        self.enrich_callees = enrich_callees
        self.callee_dropout = callee_dropout
        self.training_mode = True  # Set to False during eval to disable enrichment

        # Load tokenizer (votes or BPE)
        if votes_vocab_path and os.path.exists(votes_vocab_path):
            self.sp = VotesTokenizer(vocab_path=votes_vocab_path)
            self.tokenizer_type = 'votes'
            print(f"Using votes tokenizer: {self.sp.get_piece_size()} tokens")
        else:
            self.sp = spm.SentencePieceProcessor(model_file=bpe_model_path)
            self.tokenizer_type = 'bpe'

        # Load external vocabulary (prefer override from checkpoint)
        if ext_vocab_override is not None:
            self.ext_vocab = ext_vocab_override
        else:
            with open(external_vocab_path) as f:
                ext_data = json.load(f)
                self.ext_vocab = ext_data['vocabulary']

        # Load match index (maps graph_file → {binary, address, bap_name, real_name})
        if match_index_override is not None:
            self.match_index = match_index_override
        else:
            with open(match_index_path) as f:
                self.match_index = json.load(f)

        print(f"Match index: {len(self.match_index)} matched functions")

        # Load all external calls, indexed by (binary, function_name)
        self.ext_calls = {}
        for ef in Path(external_calls_dir).glob('*_external.json'):
            with open(ef) as f:
                data = json.load(f)
                binary = data['binary']
                for func in data.get('functions', []):
                    func_key = func.get('function_name', func.get('function_addr', ''))
                    calls = [c['name'] for c in func.get('external_calls', [])]
                    self.ext_calls[(binary, func_key)] = calls

        # Load string references, indexed by (binary, function_name)
        self.string_refs = {}
        self.string_vocab = {'<PAD>': 0, '<UNK>': 1, '<NO_STR>': 2}
        self.max_string_tokens = 15
        if string_refs_dir and os.path.isdir(string_refs_dir):
            # Load string vocabulary
            if string_vocab_path and os.path.exists(string_vocab_path):
                with open(string_vocab_path) as f:
                    sv = json.load(f)
                    self.string_vocab = sv.get('vocabulary', self.string_vocab)
            # Load per-binary string references
            for sf in Path(string_refs_dir).glob('*_strings.json'):
                with open(sf) as f:
                    data = json.load(f)
                    binary = data.get('binary', '')
                    for func in data.get('functions', []):
                        fn = func.get('function_name', '')
                        tokens = func.get('string_tokens', [])
                        if tokens:
                            self.string_refs[(binary, fn)] = tokens
            print(f"String refs: {len(self.string_refs)} functions, "
                  f"{len(self.string_vocab)} vocab tokens")

        # First pass: load all graphs indexed by (binary, bap_name) for callee lookup
        self._all_graphs = {}  # (binary, bap_name) -> graph data
        self.samples = []
        self.token_counter = defaultdict(int)

        thunks_resolved = 0
        for graph_path, match_info in self.match_index.items():
            if not os.path.exists(graph_path):
                continue

            with open(graph_path) as f:
                graph = json.load(f)

            binary = match_info['binary']
            bap_name = match_info['bap_name']
            real_name = match_info['real_name']

            # ── Thunk resolution (O0 ENDBR64 fix) ──
            # At O0 with PIE+CET, BAP lifts most functions as 1-token thunks:
            #   endbr64; jmp addr+4  →  CALL_INTERNAL to sub_(addr+4)
            # Resolve by replacing the thunk graph with its callee's graph.
            all_tokens = [t for b in graph['blocks'] for t in b['tokens']]
            if len(all_tokens) <= 2 and 'CALL_INTERNAL' in all_tokens:
                callees = graph.get('internal_callees', [])
                if callees:
                    callee_name = callees[0]
                    graph_dir = os.path.dirname(graph_path)
                    callee_path = os.path.join(graph_dir, f"{binary}_{callee_name}.json")
                    if os.path.exists(callee_path):
                        with open(callee_path) as f:
                            callee_graph = json.load(f)
                        # Use callee's blocks/edges but keep original metadata
                        graph['blocks'] = callee_graph['blocks']
                        graph['edges'] = callee_graph['edges']
                        # Merge callee's internal_callees and inherit ext calls
                        graph['internal_callees'] = callee_graph.get('internal_callees', [])
                        thunks_resolved += 1
                        # Also look up ext calls from the callee
                        callee_ext = self.ext_calls.get((binary, callee_name), [])
                        if callee_ext:
                            self.ext_calls[(binary, bap_name)] = callee_ext

            # Count tokens for vocabulary
            for block in graph['blocks']:
                for tok in block['tokens']:
                    self.token_counter[tok] += 1

            # Index by (binary, bap_name) for callee lookup
            self._all_graphs[(binary, bap_name)] = graph

            # Look up external calls for this function
            ext = self.ext_calls.get((binary, bap_name), [])

            self.samples.append({
                'binary': binary,
                'address': match_info['address'],
                'bap_name': bap_name,
                'name': real_name,
                'graph': graph,
                'ext_calls': ext,
            })

        if thunks_resolved > 0:
            print(f"Thunk resolution: {thunks_resolved} O0 thunks resolved to real code")

        # Also index graphs by (binary, sub_XXXX address) for callee resolution
        self._graphs_by_addr = {}
        for (binary, bap_name), graph in self._all_graphs.items():
            addr = graph.get('address', '')
            if addr:
                self._graphs_by_addr[(binary, addr)] = graph

        # Build reverse call graph: (binary, callee_name) -> list of caller bap_names
        self._callers = defaultdict(list)
        for (binary, bap_name), graph in self._all_graphs.items():
            for callee_name in graph.get('internal_callees', []):
                self._callers[(binary, callee_name)].append(bap_name)

        # Build binary PLT fingerprint: aggregate all ext calls per binary
        self._binary_ext_calls = defaultdict(list)
        for (binary, func_name), calls in self.ext_calls.items():
            self._binary_ext_calls[binary].extend(calls)
        # Deduplicate and sort for determinism
        for binary in self._binary_ext_calls:
            self._binary_ext_calls[binary] = sorted(set(self._binary_ext_calls[binary]))
        self.max_binary_ext = 30  # top-30 PLT calls per binary as fingerprint

        # Build or use provided token vocabulary
        # Raise cap when enriching callees (need room for CALL_<name> tokens)
        # Limited to top-500 most common callee names to keep embedding manageable
        max_token_vocab = 3500 if self.enrich_callees else 3000
        if token_vocab is None:
            self.token_vocab = {'<PAD>': 0, '<UNK>': 1}
            for tok, count in sorted(self.token_counter.items(), key=lambda x: (-x[1], x[0])):
                if len(self.token_vocab) >= max_token_vocab:
                    break
                self.token_vocab[tok] = len(self.token_vocab)
        else:
            self.token_vocab = token_vocab

        # Build callee name enrichment mapping: (binary, bap_name) -> real_name
        self._callee_real_names = {}
        if self.enrich_callees:
            for s in self.samples:
                self._callee_real_names[(s['binary'], s['bap_name'])] = s['name']
                # Also map by address: (binary, sub_XXXX)
                addr = s['graph'].get('address', '')
                if addr:
                    alt_name = f"sub_{addr[2:]}" if addr.startswith('0x') else None
                    if alt_name:
                        self._callee_real_names[(s['binary'], alt_name)] = s['name']

            # Add CALL_<real_name> tokens to vocabulary, most frequent first
            callee_token_counts = Counter()
            for s in self.samples:
                for callee_bap in s['graph'].get('internal_callees', []):
                    real = self._callee_real_names.get((s['binary'], callee_bap))
                    if real:
                        callee_token_counts[f'CALL_{real}'] += 1
            new_call_tokens = set(callee_token_counts.keys())
            for tok, _count in callee_token_counts.most_common():
                if tok not in self.token_vocab and len(self.token_vocab) < max_token_vocab:
                    self.token_vocab[tok] = len(self.token_vocab)
            print(f"Callee enrichment: {len(new_call_tokens)} CALL_<name> tokens, "
                  f"{sum(1 for t in new_call_tokens if f'CALL_{t[5:]}' in self.token_vocab or t in self.token_vocab)} in vocab, "
                  f"dropout={self.callee_dropout}")

        print(f"Loaded {len(self.samples)} functions, "
              f"{len(self.token_vocab)} BAP-IR tokens, "
              f"{len(self.ext_vocab)} external functions")

    def _get_callee_signature(self, binary, callee_name):
        """Get a compact token signature for an internal callee function.

        Returns the first max_tokens tokens of the callee's first few blocks,
        providing a fingerprint of what the callee does.
        """
        # Try to find callee by name
        callee_graph = self._all_graphs.get((binary, callee_name))
        if callee_graph is None:
            # Try by address (sub_XXXX -> 0xXXXX)
            if callee_name.startswith('sub_'):
                addr = '0x' + callee_name[4:]
                callee_graph = self._graphs_by_addr.get((binary, addr))
        if callee_graph is None:
            return []

        # Extract first N tokens across first 3 blocks as signature
        sig_tokens = []
        for block in callee_graph['blocks'][:3]:
            sig_tokens.extend(block['tokens'][:5])
            if len(sig_tokens) >= 10:
                break
        return sig_tokens[:10]

    def _get_caller_signature(self, binary, caller_name):
        """Get a compact token signature for a caller function."""
        caller_graph = self._all_graphs.get((binary, caller_name))
        if caller_graph is None:
            if caller_name.startswith('sub_'):
                addr = '0x' + caller_name[4:]
                caller_graph = self._graphs_by_addr.get((binary, addr))
        if caller_graph is None:
            return []
        sig_tokens = []
        for block in caller_graph['blocks'][:3]:
            sig_tokens.extend(block['tokens'][:5])
            if len(sig_tokens) >= 10:
                break
        return sig_tokens[:10]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        graph = sample['graph']

        blocks = graph['blocks'][:self.max_blocks]
        edges = graph['edges']
        num_blocks = len(blocks)

        # ── Compute block degrees from edges ──
        in_degrees, out_degrees = compute_block_degrees(edges, num_blocks)

        # ── Build callee name mapping for this function (if enrichment enabled) ──
        callee_name_map = {}  # bap_name -> CALL_<real_name> token
        if self.enrich_callees and self.training_mode:
            import random as _rng
            for callee_bap in graph.get('internal_callees', []):
                # Dropout: randomly skip enrichment for some callees
                if _rng.random() < self.callee_dropout:
                    continue
                real = self._callee_real_names.get((sample['binary'], callee_bap))
                if real:
                    call_tok = f'CALL_{real}'
                    if call_tok in self.token_vocab:
                        callee_name_map[callee_bap] = self.token_vocab[call_tok]

        # ── Encode blocks + compute statistics ──
        block_tokens = []
        block_features = []
        # Track which internal callee each CALL_INTERNAL corresponds to
        callee_iter = iter(graph.get('internal_callees', []))
        for i, block in enumerate(blocks):
            # Token IDs (for transformer) — with optional callee enrichment
            token_ids = []
            for t in block['tokens'][:self.max_tokens]:
                if t == 'CALL_INTERNAL' and callee_name_map:
                    callee_bap = next(callee_iter, None)
                    if callee_bap and callee_bap in callee_name_map:
                        token_ids.append(callee_name_map[callee_bap])
                    else:
                        token_ids.append(self.token_vocab.get(t, self.token_vocab['<UNK>']))
                else:
                    if t == 'CALL_INTERNAL':
                        next(callee_iter, None)  # consume the callee even if not enriching
                    token_ids.append(self.token_vocab.get(t, self.token_vocab['<UNK>']))
            token_ids += [0] * (self.max_tokens - len(token_ids))
            block_tokens.append(token_ids)

            # Block statistics (NEW)
            features = compute_block_features(
                block_tokens=block['tokens'],
                block_idx=i,
                num_blocks=num_blocks,
                in_degree=in_degrees[i] if i < len(in_degrees) else 0,
                out_degree=out_degrees[i] if i < len(out_degrees) else 0,
            )
            block_features.append(features)

        # Pad to max_blocks
        while len(block_tokens) < self.max_blocks:
            block_tokens.append([0] * self.max_tokens)
            block_features.append([0.0] * NUM_BLOCK_FEATURES)

        # ── Edges ──
        valid_edges = [e for e in edges if e[0] < num_blocks and e[1] < num_blocks]
        if not valid_edges:
            valid_edges = [[0, 0]]

        # ── External calls ──
        ext_ids = [
            self.ext_vocab.get(name, self.ext_vocab.get('<NO_EXT>', 0))
            for name in sample['ext_calls']
        ]
        if not ext_ids:
            ext_ids = [0]

        # ── Internal callee context ──
        max_callees = 5
        max_sig_tokens = 10
        callee_sigs = []
        internal_callees = sample['graph'].get('internal_callees', [])
        for callee_name in internal_callees[:max_callees]:
            sig = self._get_callee_signature(sample['binary'], callee_name)
            sig_ids = [
                self.token_vocab.get(t, self.token_vocab['<UNK>'])
                for t in sig[:max_sig_tokens]
            ]
            sig_ids += [0] * (max_sig_tokens - len(sig_ids))
            callee_sigs.append(sig_ids)
        while len(callee_sigs) < max_callees:
            callee_sigs.append([0] * max_sig_tokens)

        # ── Caller context (who calls this function) ──
        max_callers = 5
        caller_sigs = []
        callers = self._callers.get((sample['binary'], sample['bap_name']), [])
        for caller_name in callers[:max_callers]:
            sig = self._get_caller_signature(sample['binary'], caller_name)
            sig_ids = [
                self.token_vocab.get(t, self.token_vocab['<UNK>'])
                for t in sig[:max_sig_tokens]
            ]
            sig_ids += [0] * (max_sig_tokens - len(sig_ids))
            caller_sigs.append(sig_ids)
        while len(caller_sigs) < max_callers:
            caller_sigs.append([0] * max_sig_tokens)

        # ── Binary PLT fingerprint ──
        binary_ext = self._binary_ext_calls.get(sample['binary'], [])
        binary_ext_ids = [
            self.ext_vocab.get(name, self.ext_vocab.get('<NO_EXT>', 0))
            for name in binary_ext[:self.max_binary_ext]
        ]
        binary_ext_ids += [0] * (self.max_binary_ext - len(binary_ext_ids))

        # ── Target name ──
        name = sample['name']
        if self.tokenizer_type == 'votes':
            name_ids = self.sp.encode(name)
        else:
            name_split = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
            name_split = name_split.replace('_', ' _ ').replace('.', ' . ').lower()
            name_ids = self.sp.encode(name_split)
        name_ids = name_ids[:self.max_name_len]

        sos_id = self.sp.bos_id()
        eos_id = self.sp.eos_id()
        decoder_input = [sos_id] + name_ids
        decoder_target = name_ids + [eos_id]

        # ── String references ──
        str_tokens = self.string_refs.get((sample['binary'], sample['bap_name']), [])
        str_ids = [
            self.string_vocab.get(t, self.string_vocab.get('<UNK>', 1))
            for t in str_tokens[:self.max_string_tokens]
        ]
        str_ids += [0] * (self.max_string_tokens - len(str_ids))

        return {
            'block_tokens': torch.tensor(block_tokens, dtype=torch.long),
            'block_features': torch.tensor(block_features, dtype=torch.float32),
            'num_blocks': num_blocks,
            'edge_index': torch.tensor(valid_edges, dtype=torch.long).t().contiguous(),
            'ext_call_ids': torch.tensor(ext_ids, dtype=torch.long),
            'callee_tokens': torch.tensor(callee_sigs, dtype=torch.long),
            'caller_tokens': torch.tensor(caller_sigs, dtype=torch.long),
            'binary_ext_ids': torch.tensor(binary_ext_ids, dtype=torch.long),
            'num_callees': min(len(internal_callees), max_callees),
            'num_callers': min(len(callers), max_callers),
            'string_tokens': torch.tensor(str_ids, dtype=torch.long),
            'decoder_input': torch.tensor(decoder_input, dtype=torch.long),
            'decoder_target': torch.tensor(decoder_target, dtype=torch.long),
            'name': name,
            'binary': sample['binary'],
            'address': sample['address'],
        }

    def get_splits(self, train_ratio=0.7, val_ratio=0.15,
                    split_file='data/split_assignments.json'):
        """Split by binary to avoid data leakage.

        If split_file exists, use fixed binary assignments (safe for data expansion).
        Otherwise, fall back to seeded random shuffle (legacy behavior).
        New binaries not in the split file default to train.
        """
        if os.path.exists(split_file):
            with open(split_file) as f:
                assignments = json.load(f)
            val_bins_set = set(assignments['val'])
            test_bins_set = set(assignments['test'])
            train_bins_set = set(assignments['train'])

            # Excluded binaries are dropped entirely
            excluded_set = set(assignments.get('excluded', []))

            # Any new binary not in the file goes to train (unless excluded)
            all_binaries = set(s['binary'] for s in self.samples)
            known = train_bins_set | val_bins_set | test_bins_set | excluded_set
            new_binaries = all_binaries - known
            if new_binaries:
                train_bins_set.update(new_binaries)
                print(f"New binaries assigned to train: {sorted(new_binaries)}")
            if excluded_set & all_binaries:
                print(f"Excluded binaries: {len(excluded_set & all_binaries)}")

            train_idx = [i for i, s in enumerate(self.samples) if s['binary'] in train_bins_set]
            val_idx = [i for i, s in enumerate(self.samples) if s['binary'] in val_bins_set]
            test_idx = [i for i, s in enumerate(self.samples) if s['binary'] in test_bins_set]

            print(f"Splits (fixed): {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")
            print(f"Binaries: {len(train_bins_set & all_binaries)} train, "
                  f"{len(val_bins_set & all_binaries)} val, {len(test_bins_set & all_binaries)} test")
        else:
            # Legacy: seeded random shuffle
            binaries = sorted(set(s["binary"] for s in self.samples))
            random.seed(42)
            random.shuffle(binaries)

            n = len(binaries)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            train_bins = set(binaries[:n_train])
            val_bins = set(binaries[n_train:n_train + n_val])
            test_bins = set(binaries[n_train + n_val:])

            train_idx = [i for i, s in enumerate(self.samples) if s['binary'] in train_bins]
            val_idx = [i for i, s in enumerate(self.samples) if s['binary'] in val_bins]
            test_idx = [i for i, s in enumerate(self.samples) if s['binary'] in test_bins]

            print(f"Splits (random): {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")
            print(f"Binaries: {len(train_bins)} train, {len(val_bins)} val, {len(test_bins)} test")

        return train_idx, val_idx, test_idx
