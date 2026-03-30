"""
Extract string references from binary functions.

For each function in a BIR file, finds constants that point to .rodata
strings in the corresponding ELF binary. These string references are
highly informative for function naming — e.g., a function referencing
"memory exhausted" is likely an error handler.

Output: JSON file per binary with function → list of referenced strings.
Also builds a string token vocabulary across all binaries.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

from elftools.elf.elffile import ELFFile


def extract_rodata_strings(binary_path):
    """Extract all strings from .rodata section of an ELF binary.

    Returns:
        rodata_base: start address of .rodata
        rodata_end: end address of .rodata
        string_map: dict {address: string} for all null-terminated strings >= 2 chars
    """
    with open(binary_path, 'rb') as f:
        elf = ELFFile(f)
        rodata = elf.get_section_by_name('.rodata')
        if rodata is None:
            return 0, 0, {}

        rodata_data = rodata.data()
        rodata_base = rodata['sh_addr']
        rodata_end = rodata_base + len(rodata_data)

        # Build address → string map for all null-terminated strings
        string_map = {}
        start = 0
        for i in range(len(rodata_data)):
            if rodata_data[i] == 0:
                if i > start and i - start >= 2:
                    try:
                        s = rodata_data[start:i].decode('ascii', errors='strict')
                        if s.isprintable():
                            string_map[rodata_base + start] = s
                    except (UnicodeDecodeError, ValueError):
                        pass
                start = i + 1

        return rodata_base, rodata_end, string_map


def extract_function_strings(bir_path, rodata_base, rodata_end, string_map):
    """Parse BIR file and find which functions reference which .rodata strings.

    Returns:
        dict {function_name: list of referenced strings}
    """
    with open(bir_path) as f:
        lines = f.readlines()

    current_func = None
    func_strings = {}
    hex_pattern = re.compile(r'0x([0-9a-fA-F]+)')

    for line in lines:
        # Function boundary: "ADDR: sub funcname(...)"
        m = re.match(r'^\S+:\s+sub\s+(\S+)\(', line)
        if m:
            current_func = m.group(1)
            func_strings[current_func] = set()
            continue

        if current_func:
            for cm in hex_pattern.finditer(line):
                val = int(cm.group(1), 16)
                if rodata_base <= val < rodata_end:
                    s = string_map.get(val)
                    if s:
                        func_strings[current_func].add(s)

    return {k: sorted(v) for k, v in func_strings.items() if v}


def tokenize_string(s, max_tokens=5):
    """Convert a string reference to semantic tokens.

    Strategy: extract meaningful words from the string, lowercase.
    Handles format strings, error messages, identifiers, etc.
    """
    # Remove format specifiers like %s, %d, %lu, etc.
    s = re.sub(r'%[-+0 #]*\d*\.?\d*[hlLzjt]*[diouxXeEfFgGaAcspn%]', '', s)
    # Remove non-alphanumeric (keep spaces and underscores)
    s = re.sub(r'[^a-zA-Z0-9_ ]', ' ', s)
    # Split into words
    words = s.lower().split()
    # Filter out very short words and numbers
    words = [w for w in words if len(w) >= 2 and not w.isdigit()]
    return words[:max_tokens]


def build_string_vocab(all_string_tokens, min_count=3, max_vocab=500):
    """Build vocabulary from string tokens across all binaries."""
    counter = Counter()
    for tokens in all_string_tokens:
        counter.update(tokens)

    # Special tokens
    vocab = {'<PAD>': 0, '<UNK>': 1, '<NO_STR>': 2}

    # Add most common tokens
    for token, count in counter.most_common():
        if count < min_count:
            break
        if len(vocab) >= max_vocab:
            break
        if token not in vocab:
            vocab[token] = len(vocab)

    return vocab


def process_binary(binary_name, stripped_dir, bir_dir):
    """Process one binary: extract string references for all its functions."""
    stripped_path = os.path.join(stripped_dir, f'{binary_name}_stripped')
    bir_path = os.path.join(bir_dir, f'{binary_name}.bir')

    if not os.path.exists(stripped_path) or not os.path.exists(bir_path):
        return {}

    rodata_base, rodata_end, string_map = extract_rodata_strings(stripped_path)
    if not string_map:
        return {}

    return extract_function_strings(bir_path, rodata_base, rodata_end, string_map)


def main():
    parser = argparse.ArgumentParser(description='Extract string references from binaries')
    parser.add_argument('--stripped-dir', default='data/stripped',
                        help='Directory containing stripped ELF binaries')
    parser.add_argument('--bir-dir', default='data/bir',
                        help='Directory containing BIR files')
    parser.add_argument('--output-dir', default='data/string_refs',
                        help='Output directory for string reference JSON files')
    parser.add_argument('--vocab-path', default='data/string_refs/string_vocab.json',
                        help='Output path for string vocabulary')
    parser.add_argument('--min-count', type=int, default=3,
                        help='Minimum token count for vocabulary')
    parser.add_argument('--max-vocab', type=int, default=500,
                        help='Maximum vocabulary size')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Discover all BIR files
    bir_files = sorted([f for f in os.listdir(args.bir_dir) if f.endswith('.bir')])
    print(f"Found {len(bir_files)} BIR files")

    all_string_tokens = []
    total_funcs = 0
    total_with_strings = 0

    for bir_file in bir_files:
        binary_name = bir_file.replace('.bir', '')
        func_strings = process_binary(binary_name, args.stripped_dir, args.bir_dir)

        if not func_strings:
            continue

        # Save per-binary string references
        output = {
            'binary': binary_name,
            'functions': []
        }
        for func_name, strings in sorted(func_strings.items()):
            tokens = []
            for s in strings:
                tokens.extend(tokenize_string(s))
            output['functions'].append({
                'function_name': func_name,
                'strings': strings,
                'string_tokens': tokens,
            })
            all_string_tokens.append(tokens)

        output_path = os.path.join(args.output_dir, f'{binary_name}_strings.json')
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        n_funcs_total = len(func_strings)
        # Count from BIR how many total functions exist
        with open(os.path.join(args.bir_dir, bir_file)) as bf:
            n_total = sum(1 for line in bf if re.match(r'^\S+:\s+sub\s+\S+\(', line))

        total_funcs += n_total
        total_with_strings += len(func_strings)

        pct = 100 * len(func_strings) / max(n_total, 1)
        print(f"  {binary_name:45s} {len(func_strings):4d}/{n_total:4d} funcs with strings ({pct:5.1f}%)")

    # Build vocabulary
    vocab = build_string_vocab(all_string_tokens,
                                min_count=args.min_count,
                                max_vocab=args.max_vocab)

    vocab_output = {
        'vocabulary': vocab,
        'vocab_size': len(vocab),
        'total_functions_with_strings': total_with_strings,
        'total_functions': total_funcs,
        'coverage_pct': round(100 * total_with_strings / max(total_funcs, 1), 1),
    }
    with open(args.vocab_path, 'w') as f:
        json.dump(vocab_output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Total: {total_with_strings}/{total_funcs} functions with string refs "
          f"({100*total_with_strings/max(total_funcs,1):.1f}%)")
    print(f"String token vocab: {len(vocab)} tokens (min_count={args.min_count})")
    print(f"Saved to: {args.output_dir}/")
    print(f"Vocab: {args.vocab_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
