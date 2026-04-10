#!/usr/bin/env python3
"""
Convert Ghidra decompilation JSONs to SymGen input format for inference.

For each cross-project binary:
  1. Load Ghidra JSON (stripped decompilation)
  2. Load GT labels (address → real function name)
  3. For each function with a GT match:
     - Take the decompiled C code (which has FUN_XXXX names)
     - Replace the target function name with [MASK]
     - Create SymGen input entry with instruction/input/output

Output: JSON file compatible with SymGen's predict.py
"""
import json
import os
import sys
import re
import glob
import argparse
from collections import defaultdict


INSTRUCTION = (
    "Suppose you are an expert in software reverse engineering. "
    "Here is a piece of decompiled code, you should infer code semantics "
    "and tell me the original function name from the contents of the "
    "function to replace [MASK]. And you need to tell me your answer."
)

# Cross-project eval binaries (O0/O2 only, matching FuncR eval)
EVAL_BINARIES = [
    ("diffutils", "diff", [""]),
    ("diffutils", "cmp", [""]),
    ("diffutils", "sdiff", [""]),
    ("diffutils", "diff3", [""]),
    ("curl", "curl", ["O0", "O2"]),
    ("nginx", "nginx", ["O0", "O2"]),
    ("csplit2", "cflow", ["O0", "O2"]),
    ("datamash", "datamash", ["O0", "O2"]),
    ("rcs", "rcs", ["O0", "O2"]),
    ("cppi", "cppi", ["O0", "O2"]),
    ("hello", "hello", ["O0", "O2"]),
    ("tree", "tree", ["O0", "O2"]),
    ("dos2unix", "dos2unix", ["O0", "O2"]),
    ("dos2unix", "unix2dos", ["O0", "O2"]),
    ("bzip2", "bzip2", ["O0", "O2"]),
]


def load_ground_truth(bin_name, labels_dirs):
    """Load GT labels: address → function name."""
    for ldir in labels_dirs:
        label_file = os.path.join(ldir, f"{bin_name}_labels.json")
        if os.path.exists(label_file):
            with open(label_file) as f:
                labels = json.load(f)
            gt = {}
            funcs = labels.get('functions', labels)
            if isinstance(funcs, dict):
                for name, addr in funcs.items():
                    if isinstance(addr, str):
                        # Normalize address
                        addr_clean = addr.lower().replace('0x', '').lstrip('0') or '0'
                        gt[addr_clean] = name
            return gt
    return {}


def find_ghidra_json(bin_name, decomp_dir):
    """Find the Ghidra decompilation JSON for a binary."""
    # Try exact match first
    path = os.path.join(decomp_dir, f"{bin_name}.json")
    if os.path.exists(path):
        return path
    # Try with _stripped suffix
    path = os.path.join(decomp_dir, f"{bin_name}_stripped.json")
    if os.path.exists(path):
        return path
    return None


def mask_function_name(decomp_code, func_name):
    """Replace the function name with [MASK] in decompiled code.

    Ghidra names functions as FUN_XXXXX. We replace occurrences of this name
    with [MASK] — specifically the definition (first occurrence).
    """
    if not decomp_code or not func_name:
        return decomp_code

    # Replace function name at definition point (usually first occurrence)
    # Be careful to replace the name, not substrings of other names
    # Use word boundary matching
    masked = re.sub(r'\b' + re.escape(func_name) + r'\b', '[MASK]', decomp_code, count=1)
    return masked


def process_binary(bin_name, pkg, decomp_dir, labels_dirs):
    """Process a single binary: match Ghidra functions with GT labels."""
    ghidra_path = find_ghidra_json(bin_name, decomp_dir)
    if not ghidra_path:
        print(f"  SKIP {bin_name}: no Ghidra JSON found")
        return []

    gt = load_ground_truth(bin_name, labels_dirs)
    if not gt:
        print(f"  SKIP {bin_name}: no GT labels")
        return []

    with open(ghidra_path) as f:
        ghidra_data = json.load(f)

    entries = []
    matched = 0
    skipped_short = 0
    skipped_no_code = 0

    for addr, func_data in ghidra_data.items():
        # Normalize Ghidra address to match GT format
        addr_clean = addr.lower().replace('0x', '').lstrip('0') or '0'

        true_name = gt.get(addr_clean)
        if not true_name:
            continue

        decomp_code = func_data.get('decomp_code', '')
        if not decomp_code or decomp_code.strip() == '':
            skipped_no_code += 1
            continue

        func_name = func_data.get('func_name', '')
        assembly = func_data.get('assembly', [])

        # Skip very short functions (< 5 instructions) — same as SymGen
        if len(assembly) < 5:
            skipped_short += 1
            continue

        # Skip very long functions (> 510 instructions) — same as SymGen
        if len(assembly) > 510:
            continue

        # Mask the function name in decompiled code
        masked_code = mask_function_name(decomp_code, func_name)

        entry = {
            'instruction': INSTRUCTION,
            'input': masked_code,
            'output': f"The predicted function name is {true_name}",
            # Metadata for evaluation
            '_address': addr,
            '_binary': bin_name,
            '_package': pkg,
            '_true_name': true_name,
            '_ghidra_name': func_name,
        }
        entries.append(entry)
        matched += 1

    print(f"  {bin_name}: {matched} matched, {skipped_short} too short, "
          f"{skipped_no_code} no decomp, {len(gt)} GT total")
    return entries


def main():
    parser = argparse.ArgumentParser(description='Prepare SymGen input from Ghidra decompilations')
    parser.add_argument('--decomp-dir', default='data/symgen_decomp',
                        help='Directory with Ghidra decompilation JSONs')
    parser.add_argument('--output', default='data/symgen_input.json',
                        help='Output file path')
    parser.add_argument('--output-metadata', default='data/symgen_input_metadata.json',
                        help='Output metadata (for evaluation matching)')
    args = parser.parse_args()

    labels_dirs = ['data/labels', 'demo/labels']
    decomp_dir = args.decomp_dir

    print("=" * 60)
    print("Preparing SymGen input from Ghidra decompilations")
    print(f"Decomp dir: {decomp_dir}")
    print("=" * 60)

    all_entries = []
    per_pkg_counts = defaultdict(int)

    for pkg, binary, opt_levels in EVAL_BINARIES:
        for opt in opt_levels:
            bin_name = f"{pkg}_{binary}_{opt}" if opt else f"{pkg}_{binary}"
            entries = process_binary(bin_name, pkg, decomp_dir, labels_dirs)
            all_entries.extend(entries)
            per_pkg_counts[pkg] += len(entries)

    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total entries: {len(all_entries)}")
    print(f"\nPer-package:")
    for pkg in sorted(per_pkg_counts.keys()):
        print(f"  {pkg}: {per_pkg_counts[pkg]}")

    # Save SymGen input (instruction/input/output only — no metadata)
    symgen_entries = [
        {'instruction': e['instruction'], 'input': e['input'], 'output': e['output']}
        for e in all_entries
    ]
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(symgen_entries, f, indent=2)
    print(f"\nSymGen input saved to {args.output} ({len(symgen_entries)} entries)")

    # Save metadata for evaluation matching
    metadata = [
        {
            'address': e['_address'],
            'binary': e['_binary'],
            'package': e['_package'],
            'true_name': e['_true_name'],
            'ghidra_name': e['_ghidra_name'],
        }
        for e in all_entries
    ]
    with open(args.output_metadata, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {args.output_metadata}")


if __name__ == '__main__':
    main()
