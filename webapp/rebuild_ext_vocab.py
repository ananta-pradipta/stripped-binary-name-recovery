"""
Reconstruct the training-time external vocabulary from per-binary
external call files.

The external_vocab.json currently on disk has 120 tokens, but the
models were trained with 605. This script rebuilds the correct vocab
from the per-binary *_external.json files in data/external_calls/.

Usage:
    python rebuild_ext_vocab.py

If per-binary files don't exist, run this in the main project directory
where data/external_calls/*_external.json files are present.
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path


def rebuild_from_external_files(ext_calls_dir: str) -> dict:
    """Rebuild vocab from per-binary *_external.json files."""
    counter = Counter()
    file_count = 0

    for ef in sorted(Path(ext_calls_dir).glob("*_external.json")):
        with open(ef) as f:
            data = json.load(f)
        for func in data.get("functions", []):
            for call in func.get("external_calls", []):
                counter[call["name"]] += 1
        file_count += 1

    if file_count == 0:
        return None

    vocab = {"<NO_EXT>": 0}
    idx = 1
    for name, count in counter.most_common():
        vocab[name] = idx
        idx += 1

    print(f"Rebuilt vocab from {file_count} binary files")
    print(f"Vocab size: {len(vocab)} (including <NO_EXT>)")
    return vocab


def rebuild_from_match_index_and_graphs(
    match_index_path: str, graphs_dir: str
) -> dict:
    """
    Fallback: rebuild vocab from graph files that contain CALL_xxx tokens.
    This is less accurate but better than the 120-token version.
    """
    counter = Counter()

    if not os.path.exists(match_index_path):
        return None

    with open(match_index_path) as f:
        match_index = json.load(f)

    for graph_path in match_index:
        if not os.path.exists(graph_path):
            continue
        with open(graph_path) as f:
            graph = json.load(f)
        for block in graph.get("blocks", []):
            for token in block.get("tokens", []):
                if token.startswith("CALL_") and token not in (
                    "CALL_INTERNAL", "CALL_INDIRECT"
                ):
                    # Extract the external function name
                    ext_name = token[5:]  # Remove "CALL_" prefix
                    counter[ext_name] += 1

    if not counter:
        return None

    vocab = {"<NO_EXT>": 0}
    idx = 1
    for name, count in counter.most_common():
        vocab[name] = idx
        idx += 1

    print(f"Rebuilt vocab from graph CALL_ tokens")
    print(f"Vocab size: {len(vocab)} (including <NO_EXT>)")
    return vocab


def main():
    # Try multiple locations
    search_dirs = [
        "data/external_calls",
        "../data/external_calls",
        os.path.expanduser("~/cs785-project/data/external_calls"),
    ]

    vocab = None

    # Method 1: From per-binary external call files
    for ext_dir in search_dirs:
        if os.path.isdir(ext_dir):
            ext_files = list(Path(ext_dir).glob("*_external.json"))
            if ext_files:
                print(f"Found {len(ext_files)} external call files in {ext_dir}")
                vocab = rebuild_from_external_files(ext_dir)
                if vocab:
                    break

    # Method 2: From graph CALL_ tokens
    if vocab is None:
        print("No external call files found. Trying graph files...")
        match_paths = [
            "data/match_index.json",
            "../data/match_index.json",
            os.path.expanduser("~/cs785-project/data/match_index.json"),
        ]
        for mp in match_paths:
            if os.path.exists(mp):
                vocab = rebuild_from_match_index_and_graphs(mp, "data/graphs")
                if vocab:
                    break

    if vocab is None:
        print("\nERROR: Could not rebuild vocabulary.")
        print("Please ensure one of the following exists:")
        print("  1. data/external_calls/*_external.json (per-binary ext call files)")
        print("  2. data/match_index.json + data/graphs/*.json (training graph data)")
        print("\nThese were generated during Step 2/3 of the training pipeline.")
        sys.exit(1)

    # Save
    output_path = "data/external_calls/external_vocab.json"
    backup_path = "data/external_calls/external_vocab_120.json.bak"

    # Backup old vocab
    if os.path.exists(output_path):
        with open(output_path) as f:
            old_data = json.load(f)
        old_size = old_data.get("vocab_size", "?")
        print(f"\nBacking up current vocab ({old_size} tokens) to {backup_path}")
        with open(backup_path, "w") as f:
            json.dump(old_data, f, indent=2)

    # Write new vocab
    output_data = {"vocab_size": len(vocab), "vocabulary": vocab}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved rebuilt vocab to {output_path}")
    print(f"Vocab size: {len(vocab)} tokens")

    # Verify against expected size
    if len(vocab) == 605:
        print("Matches expected training-time size (605). Good to go!")
    elif len(vocab) > 100:
        print(f"Note: expected 605 from training, got {len(vocab)}.")
        print("This may be close enough if some binaries were removed.")
    else:
        print(f"WARNING: Only {len(vocab)} tokens. This seems too small.")
        print("The models were trained with 605 tokens.")


if __name__ == "__main__":
    main()
