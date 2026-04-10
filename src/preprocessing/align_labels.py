"""
Step 2: Align ground truth labels from debug binary.

UPDATED: BAP recovers actual function names (not sub_XXXX) from
stripped binaries when the PLT/GOT provides them. We extract labels
from the debug binary and match by function name where possible,
by address otherwise.
"""
import json
import subprocess
import re
import argparse
import os

# Patterns to discard
DISCARD_PATTERNS = [
    r'^_start$', r'^__libc_', r'^_init$', r'^_fini$',
    r'^__do_global_', r'^frame_dummy$', r'^register_tm_clones$',
    r'^deregister_tm_clones$', r'^__x86\.', r'^\.',
    r'^_GLOBAL_', r'^__cxa_', r'^__stack_chk',
    r'^__assert_fail$', r'^__ctype_', r'^__errno_',
    r'^__overflow$', r'^\.plt', r'^\.init', r'^\.fini',
]
DISCARD_RE = [re.compile(p) for p in DISCARD_PATTERNS]


def should_discard(name: str) -> bool:
    """Check if function name should be discarded."""
    if not name or len(name) < 2:
        return True
    return any(r.search(name) for r in DISCARD_RE)


def extract_symbols(binary_path: str) -> dict:
    """
    Extract function symbols from a debug binary using nm.
    Returns dict mapping function_name -> hex address
    """
    result = subprocess.run(
        ['nm', '-n', '--defined-only', binary_path],
        capture_output=True, text=True
    )

    symbols = {}
    addr_to_name = {}

    for line in result.stdout.strip().split('\n'):
        parts = line.split()
        if len(parts) >= 3 and parts[1].upper() == 'T':
            addr = '0x' + parts[0]
            name = parts[2]
            if not should_discard(name):
                symbols[name] = addr
                addr_to_name[addr] = name

    return symbols, addr_to_name


def save_labels(symbols: dict, binary_name: str, output_dir: str):
    """Save labels as JSON."""
    os.makedirs(output_dir, exist_ok=True)
    output = {
        'binary': binary_name,
        'num_functions': len(symbols),
        'functions': symbols,  # name -> address mapping
        'name_to_addr': symbols,
        'addr_to_name': {v: k for k, v in symbols.items()},
    }
    out_path = os.path.join(output_dir, f'{binary_name}_labels.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  Saved {len(symbols)} labels to {out_path}")

    # Show sample
    sample = list(symbols.items())[:10]
    for name, addr in sample:
        print(f"    {name:30s} @ {addr}")
    if len(symbols) > 10:
        print(f"    ... and {len(symbols) - 10} more")


def main():
    parser = argparse.ArgumentParser(description="Extract ground truth labels")
    parser.add_argument('--debug-binary', required=True)
    parser.add_argument('--binary-name', required=True)
    parser.add_argument('--output-dir', default='data/labels')
    args = parser.parse_args()

    print(f"Extracting labels from {args.debug_binary}...")
    symbols, addr_to_name = extract_symbols(args.debug_binary)
    save_labels(symbols, args.binary_name, args.output_dir)


if __name__ == '__main__':
    main()
