"""FunctionDatasetV2 — dataset v2 loader over parse_bir_v3 graphs + matcher-v2 index.

Same sample interface as FunctionDataset (block_tokens / edge_index / ext_call_ids /
callee_tokens / caller_tokens / binary_ext_ids / string_tokens / decoder_* / name /
binary / address) so the CCS model, collate and training loop run unchanged; only the
corpus construction differs:

  * records come from data/match_index_v2.json (address-matched, thunk-deduplicated,
    aliases + binding + in_dynsym per record — see scripts/build_match_index_v2.py);
  * graphs come from data/graphs_v3/ (real CFG edges, RETURN tokens, call kinds);
  * external calls come from the graph's own call sites (kind == import, call order),
    no *_external.json;
  * internal callees = internal_sub ∪ internal_named callees; thunk aliases
    (record['dropped_graphs']) resolve to the kept body graph;
  * strings come from data/string_refs_v2/ (function-specific, from the stripped ELF);
  * every record carries `in_dynsym` / `binding` / `tok_hash` for strata and dedup;
  * split guard: get_splits() (inherited, B8) refuses unassigned binaries.

Filtering knobs (all recorded in `self.filter_stats`):
  min_tokens (default 1), exclude_binaries (set), only_binaries (set), corpora (set).
"""
import json
import os
import re
from collections import Counter, defaultdict
from typing import Optional

from src.preprocessing.build_dataset import FunctionDataset

STR_TOKEN_RE = re.compile(r'[A-Za-z]+|\d+')


def tokenize_string(s: str, max_tokens: int = 8):
    toks = []
    for w in STR_TOKEN_RE.findall(s):
        w = w.lower()
        # split camelCase remnants are already lower-cased by findall on [A-Za-z]+; keep short words
        if len(w) >= 2:
            toks.append(w)
        if len(toks) >= max_tokens:
            break
    return toks


class FunctionDatasetV2(FunctionDataset):
    def __init__(self,
                 match_index_path: str = 'data/match_index_v2.json',
                 string_refs_dir: Optional[str] = 'data/string_refs_v2',
                 votes_vocab_path: Optional[str] = None,
                 bpe_model_path: Optional[str] = None,
                 ext_vocab: Optional[dict] = None,
                 token_vocab: Optional[dict] = None,
                 string_vocab: Optional[dict] = None,
                 max_blocks: int = 30, max_tokens: int = 20, max_name_len: int = 20,
                 min_tokens: int = 1,
                 exclude_binaries: Optional[set] = None,
                 only_binaries: Optional[set] = None,
                 corpora: Optional[set] = None,
                 max_token_vocab: int = 3000,
                 max_ext_vocab: int = 5000,
                 max_string_vocab: int = 5000,
                 vocab_binaries: Optional[set] = None,
                 quiet: bool = False):
        # NOTE: deliberately not calling FunctionDataset.__init__ (legacy corpus loader).
        self.max_blocks = max_blocks
        self.max_tokens = max_tokens
        self.max_name_len = max_name_len
        self.enrich_callees = False
        self.callee_dropout = 0.0
        self.training_mode = True
        self.max_string_tokens = 15
        self.max_binary_ext = 30
        self.filter_stats = Counter()

        # ---- name tokenizer (Votes preferred; same as v1)
        if votes_vocab_path:
            from src.preprocessing.build_votes import VotesTokenizer
            self.sp = VotesTokenizer(vocab_path=votes_vocab_path)
            self.tokenizer_type = 'votes'
        elif bpe_model_path:
            import sentencepiece as spm
            self.sp = spm.SentencePieceProcessor(model_file=bpe_model_path)
            self.tokenizer_type = 'bpe'
        else:
            raise ValueError('need votes_vocab_path or bpe_model_path')

        # ---- records
        with open(match_index_path) as fh:
            records = json.load(fh)
        self.records = []
        for r in records:
            if corpora and r.get('corpus') not in corpora:
                self.filter_stats['corpus'] += 1; continue
            if only_binaries is not None and r['binary'] not in only_binaries:
                self.filter_stats['only_binaries'] += 1; continue
            if exclude_binaries and r['binary'] in exclude_binaries:
                self.filter_stats['exclude_binaries'] += 1; continue
            if r['n_tokens'] < min_tokens:
                self.filter_stats['min_tokens'] += 1; continue
            self.records.append(r)

        # ---- load graphs, build indices
        self._all_graphs = {}
        self._graphs_by_addr = {}
        self.ext_calls = {}
        self.samples = []
        self.token_counter = defaultdict(int)
        ext_counter = Counter()
        for r in self.records:
            gp = r['graph']
            if not os.path.exists(gp):
                self.filter_stats['graph_missing'] += 1; continue
            with open(gp) as fh:
                graph = json.load(fh)
            binary, bap_name = r['binary'], r['bap_name']
            # external calls in call order (unique), from import call sites
            ext = []
            for cs in graph.get('call_sites', []):
                if cs['kind'] == 'import' and cs['name'] not in ext:
                    ext.append(cs['name'])
            # internal callees: sub_ + named (thunk aliases resolved at lookup time)
            graph['internal_callees'] = sorted(set(graph.get('internal_callees', [])) |
                                               set(graph.get('internal_named_callees', [])))
            for b in graph['blocks']:
                for t in b['tokens']:
                    self.token_counter[t] += 1
            self._all_graphs[(binary, bap_name)] = graph
            for alias in r.get('dropped_graphs', []):
                self._all_graphs.setdefault((binary, alias), graph)
            if graph.get('entry_addr'):
                self._graphs_by_addr[(binary, graph['entry_addr'])] = graph
                self._graphs_by_addr.setdefault((binary, r['label_addr']), graph)
            self.ext_calls[(binary, bap_name)] = ext
            ext_counter.update(ext)
            self.samples.append({
                'binary': binary, 'address': r['entry_addr'], 'bap_name': bap_name,
                'name': r['real_name'], 'aliases': r.get('aliases', []), 'graph': graph, 'ext_calls': ext,
                'in_dynsym': r.get('in_dynsym', False), 'binding': r.get('binding', ''),
                'tok_hash': r.get('tok_hash', ''), 'corpus': r.get('corpus', ''),
                'aliases_graphs': r.get('dropped_graphs', []),
            })

        # ---- reverse call graph
        self._callers = defaultdict(list)
        for (binary, bap_name), graph in self._all_graphs.items():
            for callee_name in graph.get('internal_callees', []):
                self._callers[(binary, callee_name)].append(bap_name)
        for k in self._callers:
            self._callers[k] = sorted(set(self._callers[k]))
        # thunk aliases: callers of the alias (sub_X thunk) are callers of the body (sub_X+4)
        for s in self.samples:
            for alias in s['aliases_graphs']:
                extra = self._callers.get((s['binary'], alias))
                if extra:
                    merged = set(self._callers.get((s['binary'], s['bap_name']), [])) | set(extra)
                    merged.discard(s['bap_name']); merged.discard(alias)
                    self._callers[(s['binary'], s['bap_name'])] = sorted(merged)

        # ---- binary PLT fingerprint
        self._binary_ext_calls = defaultdict(set)
        for (binary, _), calls in self.ext_calls.items():
            self._binary_ext_calls[binary].update(calls)
        self._binary_ext_calls = {b: sorted(v) for b, v in self._binary_ext_calls.items()}

        # ---- vocabularies (built from vocab_binaries subset if given — the train split)
        def _restrict(counter_fn):
            if vocab_binaries is None:
                return None
            c = Counter()
            for s in self.samples:
                if s['binary'] in vocab_binaries:
                    counter_fn(c, s)
            return c
        if token_vocab is None:
            tc = _restrict(lambda c, s: c.update(t for b in s['graph']['blocks'] for t in b['tokens']))
            tc = tc if tc is not None else self.token_counter
            self.token_vocab = {'<PAD>': 0, '<UNK>': 1}
            for tok, _ in sorted(tc.items(), key=lambda x: (-x[1], x[0])):
                if len(self.token_vocab) >= max_token_vocab:
                    break
                self.token_vocab[tok] = len(self.token_vocab)
        else:
            self.token_vocab = token_vocab
        if ext_vocab is None:
            ec = _restrict(lambda c, s: c.update(s['ext_calls']))
            ec = ec if ec is not None else ext_counter
            self.ext_vocab = {'<PAD>': 0, '<NO_EXT>': 1, '<UNK>': 2}
            for name, _ in sorted(ec.items(), key=lambda x: (-x[1], x[0])):
                if len(self.ext_vocab) >= max_ext_vocab:
                    break
                self.ext_vocab[name] = len(self.ext_vocab)
        else:
            self.ext_vocab = ext_vocab

        # ---- strings
        self.string_refs = {}
        str_counter = Counter()
        if string_refs_dir and os.path.isdir(string_refs_dir):
            wanted = defaultdict(set)
            for s in self.samples:
                wanted[s['binary']].add(s['bap_name'])
            for binary in wanted:
                p = os.path.join(string_refs_dir, binary + '.json')
                if not os.path.exists(p):
                    continue
                d = json.load(open(p))['functions']
                for bap_name in wanted[binary]:
                    strs = d.get(bap_name)
                    if strs:
                        toks = []
                        for st in strs:
                            toks.extend(tokenize_string(st))
                        self.string_refs[(binary, bap_name)] = toks[:self.max_string_tokens]
                        if vocab_binaries is None or binary in vocab_binaries:
                            str_counter.update(toks[:self.max_string_tokens])
        if string_vocab is None:
            self.string_vocab = {'<PAD>': 0, '<UNK>': 1, '<NO_STR>': 2}
            for tok, _ in sorted(str_counter.items(), key=lambda x: (-x[1], x[0])):
                if len(self.string_vocab) >= max_string_vocab:
                    break
                self.string_vocab[tok] = len(self.string_vocab)
        else:
            self.string_vocab = string_vocab

        if not quiet:
            print(f"DatasetV2: {len(self.samples)} functions from {len(set(s['binary'] for s in self.samples))} binaries; "
                  f"{len(self.token_vocab)} tokens, {len(self.ext_vocab)} ext names, {len(self.string_vocab)} string tokens; "
                  f"strings on {len(self.string_refs)} fns; in_dynsym {sum(s['in_dynsym'] for s in self.samples)}; "
                  f"filtered {dict(self.filter_stats)}")

    # record-level split policy (v3, 2026-08-23) --------------------------------------------
    def apply_split_policy(self, train_idx, val_idx, test_idx, dedup_train=True,
                           drop_body_in_train=True, drop_in_dynsym=True, quiet=False):
        """Apply the dataset-v2 record-level policy on top of the binary-level split.

        train   : keep one sample per (tok_hash, name)   [dedup_train]
        val/test: drop samples whose tok_hash occurs anywhere in train (byte-identical
                  body already supervised)                [drop_body_in_train]
                  drop in_dynsym samples (name is visible in the stripped ELF, so it is
                  not a prediction target)                [drop_in_dynsym]
        Dropped eval samples are kept in self.policy_dropped[tier] so the "seen-body" and
        "symbol-visible" strata can still be reported separately. Stats in self.policy_stats.
        """
        S = self.samples
        train_hashes = {S[i]['tok_hash'] for i in train_idx}
        stats = {'train_raw': len(train_idx)}
        if dedup_train:
            seen, kept = set(), []
            for i in train_idx:
                k = (S[i]['tok_hash'], S[i]['name'])
                if k not in seen:
                    seen.add(k); kept.append(i)
            train_idx = kept
        stats['train_kept'] = len(train_idx)
        self.policy_dropped = {}
        out = []
        for tier, idx in (('val', val_idx), ('test', test_idx)):
            keep, body, dyn = [], [], []
            for i in idx:
                if drop_body_in_train and S[i]['tok_hash'] in train_hashes:
                    body.append(i)
                elif drop_in_dynsym and S[i]['in_dynsym']:
                    dyn.append(i)
                else:
                    keep.append(i)
            self.policy_dropped[tier] = {'body_in_train': body, 'in_dynsym': dyn}
            stats[tier] = {'raw': len(idx), 'drop_body_in_train': len(body), 'drop_in_dynsym': len(dyn), 'scored': len(keep)}
            out.append(keep)
        self.policy_stats = stats
        if not quiet:
            print(f"split policy v3: {stats}")
        return train_idx, out[0], out[1]

    # thunk-alias aware signature lookups ---------------------------------------------------
    def _lookup_graph(self, binary, name):
        g = self._all_graphs.get((binary, name))
        if g is None and name.startswith('sub_'):
            g = self._graphs_by_addr.get((binary, '0x' + name[4:]))
        return g

    def _get_callee_signature(self, binary, callee_name):
        g = self._lookup_graph(binary, callee_name)
        if g is None:
            return []
        sig = []
        for block in g['blocks'][:3]:
            sig.extend(block['tokens'][:5])
            if len(sig) >= 10:
                break
        return sig[:10]

    def _get_caller_signature(self, binary, caller_name):
        return self._get_callee_signature(binary, caller_name)
