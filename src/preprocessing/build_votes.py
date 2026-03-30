"""
Build votes-based tokenization vocabulary from function names.

Instead of BPE (which fragments rare sub-tokens into characters),
votes tokenization splits on _ and camelCase boundaries, keeping
each sub-token as a meaningful whole unit.

Input:  data/match_index.json (all function names)
Output: data/votes_vocab.json (token → id mapping)
"""
import json
import re
import argparse
from collections import Counter
from pathlib import Path


# Reserved token IDs (must match BPE convention for compatibility)
PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3
UNDERSCORE_ID = 4  # '_' separator token

SPECIAL_TOKENS = {
    '<PAD>': PAD_ID,
    '<SOS>': SOS_ID,
    '<EOS>': EOS_ID,
    '<UNK>': UNK_ID,
    '_': UNDERSCORE_ID,
}


def split_name_to_subtokens(name: str) -> list:
    """Split a function name into meaningful sub-tokens.

    Steps:
    1. Split on '_' (keep '_' as separator token)
    2. Split camelCase boundaries
    3. Lowercase everything

    Examples:
        'hash_table_lookup' → ['hash', '_', 'table', '_', 'lookup']
        'handleClient'      → ['handle', '_', 'client']
        'bfd_section_list'  → ['bfd', '_', 'section', '_', 'list']
    """
    # Split camelCase first
    name = re.sub(r'([a-z])([A-Z])', r'\1_\2', name)

    parts = name.split('_')
    tokens = []
    for i, part in enumerate(parts):
        if i > 0:
            tokens.append('_')
        if part:
            tokens.append(part.lower())

    return tokens


def build_votes_vocab(match_index_path: str, min_count: int = 2) -> dict:
    """Build vocabulary from function name sub-token votes.

    Args:
        match_index_path: Path to match_index.json
        min_count: Minimum occurrences to include in vocab (default: 2)

    Returns:
        dict with 'vocab' (token→id), 'id2token' (id→token), 'vocab_size', 'min_count'
    """
    with open(match_index_path) as f:
        match_index = json.load(f)

    # Count sub-token frequencies across all function names
    subtok_counter = Counter()
    all_names = set()
    for entry in match_index.values():
        name = entry['real_name']
        if name.startswith('sub_'):
            continue
        all_names.add(name)

    for name in sorted(all_names):
        tokens = split_name_to_subtokens(name)
        for t in tokens:
            if t != '_':  # '_' is already a special token
                subtok_counter[t] += 1

    # Build vocab: special tokens + voted sub-tokens (sorted for determinism)
    vocab = dict(SPECIAL_TOKENS)
    next_id = len(SPECIAL_TOKENS)

    # Add sub-tokens that meet the vote threshold, sorted by frequency (desc) then alphabetically
    voted_tokens = sorted(
        [(tok, count) for tok, count in subtok_counter.items() if count >= min_count],
        key=lambda x: (-x[1], x[0])
    )

    for tok, count in voted_tokens:
        if tok not in vocab:
            vocab[tok] = next_id
            next_id += 1

    # Tokens below threshold become character-level fallbacks
    rare_chars = set()
    for tok, count in subtok_counter.items():
        if count < min_count and tok not in vocab:
            for ch in tok:
                rare_chars.add(ch)

    for ch in sorted(rare_chars):
        if ch not in vocab:
            vocab[ch] = next_id
            next_id += 1

    # Build reverse mapping
    id2token = {v: k for k, v in vocab.items()}

    # Stats
    total_tokens = sum(subtok_counter.values())
    covered = sum(count for tok, count in subtok_counter.items() if count >= min_count)

    print(f"Function names: {len(all_names)}")
    print(f"Unique sub-tokens: {len(subtok_counter)}")
    print(f"Vocab size (threshold >= {min_count}): {len(vocab)} "
          f"({len(voted_tokens)} voted + {len(SPECIAL_TOKENS)} special + {len(rare_chars)} char fallbacks)")
    print(f"Token coverage: {100*covered/total_tokens:.1f}%")

    return {
        'vocab': vocab,
        'id2token': id2token,
        'vocab_size': len(vocab),
        'min_count': min_count,
    }


class VotesTokenizer:
    """Tokenizer using votes-based vocabulary.

    Drop-in replacement for SentencePiece in the pipeline.
    """

    def __init__(self, vocab_path: str = None, vocab_dict: dict = None):
        if vocab_dict is not None:
            self.vocab = vocab_dict['vocab']
            self.id2token = {int(k): v for k, v in vocab_dict['id2token'].items()}
        elif vocab_path is not None:
            with open(vocab_path) as f:
                data = json.load(f)
            self.vocab = data['vocab']
            self.id2token = {int(k): v for k, v in data['id2token'].items()}
        else:
            raise ValueError("Must provide vocab_path or vocab_dict")

        self.unk_id = UNK_ID

    def encode(self, name: str) -> list:
        """Encode a function name into token IDs.

        Args:
            name: Raw function name (e.g., 'hash_table_lookup')

        Returns:
            List of token IDs
        """
        subtokens = split_name_to_subtokens(name)
        ids = []
        for tok in subtokens:
            if tok in self.vocab:
                ids.append(self.vocab[tok])
            else:
                # Character-level fallback for OOV tokens
                for ch in tok:
                    ids.append(self.vocab.get(ch, self.unk_id))
        return ids

    def decode(self, ids: list) -> str:
        """Decode token IDs back into a function name.

        Args:
            ids: List of token IDs (excluding SOS/EOS)

        Returns:
            Reconstructed function name
        """
        tokens = []
        for id in ids:
            if id in (PAD_ID, SOS_ID, EOS_ID):
                continue
            tok = self.id2token.get(id, '<UNK>')
            tokens.append(tok)

        # Join tokens, collapsing character fallbacks
        name = ''.join(tokens)
        # Clean up spaces around underscores
        name = re.sub(r'\s*_\s*', '_', name)
        return name.strip('_').strip()

    def bos_id(self) -> int:
        return SOS_ID

    def eos_id(self) -> int:
        return EOS_ID

    def get_piece_size(self) -> int:
        return len(self.vocab)


def main():
    parser = argparse.ArgumentParser(description="Build votes tokenization vocabulary")
    parser.add_argument('--match-index', default='data/match_index.json')
    parser.add_argument('--output', default='data/votes_vocab.json')
    parser.add_argument('--min-count', type=int, default=2,
                        help='Minimum sub-token frequency to include in vocab')
    args = parser.parse_args()

    result = build_votes_vocab(args.match_index, args.min_count)

    # Save
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {args.output}")

    # Test tokenization
    tokenizer = VotesTokenizer(vocab_dict=result)
    examples = [
        'hash_table_lookup', 'elfcore_grok_s390_timer',
        'swap_linux_prpsinfo32_ugid16_out', 'init_reloc_cookie_rels',
        'pch_line_len', 'bfd_section_list_append', 'xmalloc',
        'coff_get_reloc_upper_bound', 'rl_set_signals',
    ]
    print("\nTokenization examples:")
    for name in examples:
        ids = tokenizer.encode(name)
        decoded = tokenizer.decode(ids)
        tokens = [tokenizer.id2token.get(i, '?') for i in ids]
        print(f"  {name:40s} → {tokens}")
        print(f"  {'':40s}   IDs: {ids}")
        print(f"  {'':40s}   Decoded: {decoded}")
        assert decoded == name, f"Round-trip failed: {name} → {decoded}"


if __name__ == '__main__':
    main()
