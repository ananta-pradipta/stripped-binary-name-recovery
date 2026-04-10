"""
Convert preprocessed binary function data into text descriptions for LM training.

V2: Uses REAL callee/caller names (not sub_XXXX) and C-code-style prompts
that match CodeGen's pre-training distribution.
"""
import json
import os
import sys
import glob
import argparse
from collections import Counter, defaultdict


def summarize_tokens(blocks, max_types=8):
    """Summarize instruction token distribution across all blocks."""
    token_counts = Counter()
    for block in blocks:
        for tok in block.get('tokens', []):
            token_counts[tok] += 1
    top_tokens = token_counts.most_common(max_types)
    parts = [f"{tok}({count})" for tok, count in top_tokens]
    return ", ".join(parts)


def detect_loops(edges, num_blocks):
    """Simple loop detection via back-edge check."""
    if not edges:
        return False
    for src, dst in edges:
        if dst <= src:
            return True
    return False


def build_text_prompt(func_data, ext_calls, callee_real_names, caller_real_names):
    """Build a C-code-style description of a binary function.

    Args:
        callee_real_names: list of REAL function names (already resolved from sub_XXXX)
        caller_real_names: list of REAL function names
    """
    blocks = func_data.get('blocks', [])
    edges = func_data.get('edges', [])
    num_blocks = func_data.get('num_blocks', len(blocks))

    # External calls (deduplicated, with counts)
    if ext_calls:
        seen = set()
        unique = []
        for c in ext_calls:
            if c not in seen:
                unique.append(c)
                seen.add(c)
        counts = Counter(ext_calls)
        call_strs = [f"{c}()" + (f"x{counts[c]}" if counts[c] > 1 else "")
                     for c in unique[:8]]
        ext_str = ", ".join(call_strs)
    else:
        ext_str = "(none)"

    # Callee names (REAL names only, filter out unresolved)
    callee_str = ", ".join(callee_real_names[:5]) if callee_real_names else "(none)"
    # Caller names
    caller_str = ", ".join(caller_real_names[:5]) if caller_real_names else "(none)"

    # Structure info
    has_loop = detect_loops(edges, num_blocks)
    total_insns = sum(len(b.get('tokens', [])) for b in blocks)
    token_summary = summarize_tokens(blocks, max_types=8)

    # Size qualifier
    size_tag = ""
    if num_blocks == 1:
        size_tag = " (trivial wrapper)"
    elif num_blocks > 15:
        size_tag = " (complex)"

    # C-code-style comment block + function signature stub
    # CodeGen is trained on this exact format → predicts the name after "void"
    prompt = f"""/* Reverse-engineered function description.
 * Library calls: {ext_str}
 * Internal calls: {callee_str}
 * Called by: {caller_str}
 * Size: {num_blocks} blocks, {total_insns} instructions{", has loops" if has_loop else ""}{size_tag}
 * Instruction mix: {token_summary}
 */
void """

    return prompt


def main():
    parser = argparse.ArgumentParser(description='Build text dataset for LM training (V2)')
    parser.add_argument('--match-index', default='data/match_index.json')
    parser.add_argument('--graphs-dir', default='data/graphs')
    parser.add_argument('--ext-calls-dir', default='data/external_calls')
    parser.add_argument('--output', default='data/text_dataset_v2.json')
    parser.add_argument('--max-functions', type=int, default=None)
    args = parser.parse_args()

    print("Loading match index...")
    with open(args.match_index) as f:
        match_index = json.load(f)
    print(f"  {len(match_index)} entries")

    # Build BAP name -> real name lookup per binary
    # This is what lets us replace sub_XXXX with real names in callee/caller context
    print("Building BAP→real name lookup...")
    bap_to_real = defaultdict(dict)  # {binary: {bap_name: real_name}}
    for graph_path, info in match_index.items():
        bap_to_real[info['binary']][info['bap_name']] = info['real_name']
    print(f"  {sum(len(v) for v in bap_to_real.values())} mappings")

    # Load external calls per binary
    print("Loading external calls...")
    ext_calls_by_func = {}
    for ext_file in glob.glob(os.path.join(args.ext_calls_dir, '*_external.json')):
        with open(ext_file) as f:
            ext_data = json.load(f)
        binary = os.path.basename(ext_file).replace('_external.json', '')
        ext_calls_by_func[binary] = {}
        for func in ext_data.get('functions', []):
            fname = func.get('function_name', '')
            calls = [c['name'] for c in func.get('external_calls', [])]
            if calls:
                ext_calls_by_func[binary][fname] = calls

    # Build callee/caller maps per binary (BAP names)
    print("Building callee/caller context...")
    callee_map = defaultdict(lambda: defaultdict(list))
    caller_map = defaultdict(lambda: defaultdict(list))

    for graph_path, info in match_index.items():
        if not os.path.exists(graph_path):
            continue
        try:
            with open(graph_path) as f:
                graph = json.load(f)
        except:
            continue
        binary = info['binary']
        bap_name = info['bap_name']
        callees = graph.get('internal_callees', [])
        callee_map[binary][bap_name] = callees
        for callee in callees:
            caller_map[binary][callee].append(bap_name)

    # Build text dataset with REAL names
    print("Building text prompts with real callee/caller names...")
    dataset = []
    skipped = 0
    count = 0
    unresolved_callees = 0
    resolved_callees = 0

    for graph_path, info in match_index.items():
        if args.max_functions and count >= args.max_functions:
            break

        if not os.path.exists(graph_path):
            skipped += 1
            continue

        try:
            with open(graph_path) as f:
                graph = json.load(f)
        except:
            skipped += 1
            continue

        binary = info['binary']
        bap_name = info['bap_name']
        real_name = info['real_name']

        ext_calls = ext_calls_by_func.get(binary, {}).get(bap_name, [])

        # Get callee BAP names and resolve to real names
        callee_bap_names = callee_map[binary].get(bap_name, [])
        callee_real_names = []
        for cbn in callee_bap_names:
            real = bap_to_real[binary].get(cbn)
            if real:
                callee_real_names.append(real)
                resolved_callees += 1
            else:
                unresolved_callees += 1

        # Same for callers
        caller_bap_names = caller_map[binary].get(bap_name, [])
        caller_real_names = []
        for crn in caller_bap_names:
            real = bap_to_real[binary].get(crn)
            if real:
                caller_real_names.append(real)

        # Deduplicate while preserving order
        seen = set()
        callee_real_names = [x for x in callee_real_names if not (x in seen or seen.add(x))]
        seen = set()
        caller_real_names = [x for x in caller_real_names if not (x in seen or seen.add(x))]

        prompt = build_text_prompt(graph, ext_calls, callee_real_names, caller_real_names)

        dataset.append({
            'prompt': prompt,
            'name': real_name,
            'binary': binary,
            'bap_name': bap_name,
        })
        count += 1

        if count % 20000 == 0:
            print(f"  {count} functions processed...")

    print(f"\nDone: {len(dataset)} prompts, {skipped} skipped")
    print(f"Callee resolution: {resolved_callees} resolved, {unresolved_callees} unresolved")

    # Save
    with open(args.output, 'w') as f:
        json.dump(dataset, f)
    print(f"Saved to {args.output}")

    # Print samples
    print("\n=== Sample prompts ===")
    for entry in dataset[:5]:
        print(f"\n--- {entry['binary']}:{entry['bap_name']} → {entry['name']} ---")
        print(entry['prompt'] + entry['name'] + "(...)")


if __name__ == '__main__':
    main()
