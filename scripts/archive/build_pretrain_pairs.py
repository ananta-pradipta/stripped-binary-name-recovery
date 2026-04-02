#!/usr/bin/env python3
"""
Build contrastive pair index for pre-training.

Groups functions by (package_program, real_name) across different optimization
levels and generates all C(n,2) pairs. For example, if function 'main' appears
in coreutils_cat_O0, coreutils_cat_O2, and coreutils_cat_O3, we get 3 pairs.

Input:  data/match_index.json
Output: data/pretrain_pairs.json — list of [graph_path_a, graph_path_b]
"""
import json
import os
import sys
from collections import defaultdict
from itertools import combinations


def extract_base_and_opt(binary_name: str):
    """Extract (package_program, opt_level) from binary name.

    Examples:
        'coreutils_cat_O0' -> ('coreutils_cat', 'O0')
        'binutils_nm-new_O2' -> ('binutils_nm-new', 'O2')
        'coreutils2_dd_O1' -> ('coreutils2_dd', 'O1')
    """
    parts = binary_name.rsplit('_', 1)
    if len(parts) == 2 and parts[1] in ('O0', 'O1', 'O2', 'O3'):
        return parts[0], parts[1]
    return binary_name, 'unknown'


def main():
    match_index_path = 'data/match_index.json'
    output_path = 'data/pretrain_pairs.json'

    if not os.path.exists(match_index_path):
        print(f"ERROR: {match_index_path} not found")
        sys.exit(1)

    with open(match_index_path) as f:
        match_index = json.load(f)

    print(f"Loaded {len(match_index)} entries from match_index.json")

    # Group by (base_program, real_name) across optimization levels
    groups = defaultdict(list)
    for graph_path, info in match_index.items():
        if not os.path.exists(graph_path):
            continue
        binary = info['binary']
        real_name = info['real_name']
        base, opt = extract_base_and_opt(binary)
        groups[(base, real_name)].append((graph_path, opt))

    # Generate pairs: all C(n,2) within each group
    pairs = []
    multi_opt_funcs = 0
    for (base, real_name), members in groups.items():
        if len(members) > 1:
            multi_opt_funcs += 1
            # Only pair across DIFFERENT optimization levels
            for (path_a, opt_a), (path_b, opt_b) in combinations(members, 2):
                if opt_a != opt_b:
                    pairs.append([path_a, path_b])

    print(f"Functions appearing at multiple opt levels: {multi_opt_funcs}")
    print(f"Total contrastive pairs: {len(pairs)}")

    # Stats
    opt_pair_counts = defaultdict(int)
    for path_a, path_b in pairs:
        _, opt_a = extract_base_and_opt(match_index[path_a]['binary'])
        _, opt_b = extract_base_and_opt(match_index[path_b]['binary'])
        key = tuple(sorted([opt_a, opt_b]))
        opt_pair_counts[key] += 1

    print("\nPairs by optimization level combination:")
    for key, count in sorted(opt_pair_counts.items()):
        print(f"  {key[0]}-{key[1]}: {count}")

    with open(output_path, 'w') as f:
        json.dump(pairs, f)
    print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    main()
