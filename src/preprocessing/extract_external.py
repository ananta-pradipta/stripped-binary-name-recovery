"""
Step 5: Extract external function calls from BAP-IR.

UPDATED to match actual BAP output format:
  - External calls: "call @NAME:external with ..."
  - Internal calls: "call @NAME with return %HEXADDR"
  - Functions: "HEXADDR: sub FUNCNAME(params)"
"""
import json
import re
import os
import argparse
from collections import Counter
from typing import Dict, List


def extract_from_bir(bir_path: str) -> Dict[str, List[str]]:
    """
    Extract external calls from BAP-IR file.

    In this BAP format:
      - "call @NAME:external" = definitely external (libc, etc.)
      - "call @NAME with ..." where NAME is a known library function = external

    Returns:
        Dict mapping function_name -> list of external call names (ordered)
    """
    # Known external function patterns (common libc/POSIX functions)
    KNOWN_EXTERNAL = {
        'malloc', 'free', 'calloc', 'realloc',
        'printf', 'fprintf', 'sprintf', 'snprintf', 'vprintf', 'vfprintf',
        'puts', 'fputs', 'putchar', 'putc', 'fputc',
        'fopen', 'fclose', 'fread', 'fwrite', 'fgets', 'fputs', 'fflush', 'fseek', 'ftell', 'rewind',
        'open', 'close', 'read', 'write', 'lseek', 'stat', 'fstat', 'lstat',
        'socket', 'bind', 'listen', 'accept', 'connect', 'send', 'recv',
        'memcpy', 'memset', 'memmove', 'memcmp',
        'strlen', 'strcmp', 'strncmp', 'strcpy', 'strncpy', 'strcat', 'strncat', 'strstr', 'strchr', 'strrchr',
        'strtol', 'strtoul', 'strtod', 'atoi', 'atol', 'atof',
        'exit', 'abort', '_exit', 'atexit',
        'getenv', 'setenv', 'unsetenv',
        'signal', 'raise', 'sigaction',
        'fork', 'exec', 'execve', 'execvp', 'wait', 'waitpid',
        'pipe', 'dup', 'dup2',
        'mmap', 'munmap', 'mprotect', 'brk', 'sbrk',
        'pthread_create', 'pthread_join', 'pthread_mutex_lock', 'pthread_mutex_unlock',
        'getopt', 'getopt_long',
        'qsort', 'bsearch',
        'time', 'clock', 'gettimeofday', 'localtime', 'strftime',
        'isatty', 'tcgetattr', 'tcsetattr',
        'setlocale', 'bindtextdomain', 'textdomain', 'gettext', 'dcgettext', 'ngettext',
        'nl_langinfo', 'iconv', 'iconv_open', 'iconv_close',
        'error', 'error_at_line', 'perror', 'strerror',
        '__assert_fail', '__stack_chk_fail', '__overflow',
        '__errno_location', '__ctype_b_loc', '__ctype_get_mb_cur_max',
        '__cxa_atexit', '__cxa_finalize',
    }

    func_pattern = re.compile(r'^[0-9a-fA-F]+:\s+sub\s+(\S+)\(')
    call_pattern = re.compile(r'call\s+@(\w+)')
    external_marker = re.compile(r'call\s+@(\w+):external')

    functions = {}
    current_func = None
    current_calls = []

    with open(bir_path, 'r') as f:
        for line in f:
            line = line.strip()

            # New function
            func_match = func_pattern.match(line)
            if func_match:
                if current_func:
                    functions[current_func] = current_calls
                current_func = func_match.group(1)
                current_calls = []
                continue

            # External call (explicitly marked)
            ext_match = external_marker.search(line)
            if ext_match and current_func:
                name = ext_match.group(1)
                current_calls.append(name)
                continue

            # Call to known external function (not explicitly marked)
            call_match = call_pattern.search(line)
            if call_match and current_func:
                name = call_match.group(1)
                if name in KNOWN_EXTERNAL:
                    current_calls.append(name)
                continue

    if current_func:
        functions[current_func] = current_calls

    return functions


def build_vocabulary(all_calls: Dict[str, List[str]], min_count: int = 1) -> dict:
    """Build external function vocabulary from all extracted calls."""
    counter = Counter()
    for calls in all_calls.values():
        counter.update(calls)

    vocab = {"<NO_EXT>": 0}
    idx = 1
    for name, count in counter.most_common():
        if count >= min_count:
            vocab[name] = idx
            idx += 1

    return vocab


def save_external_calls(calls: dict, binary_name: str, output_dir: str):
    """Save per-function external call lists."""
    os.makedirs(output_dir, exist_ok=True)

    total_with = 0
    total_without = 0
    output = {
        'binary': binary_name,
        'functions': []
    }
    for func_name, call_list in calls.items():
        output['functions'].append({
            'function_name': func_name,
            'external_calls': [
                {'name': name, 'call_order': i}
                for i, name in enumerate(call_list)
            ],
            'num_external_calls': len(call_list),
        })
        if call_list:
            total_with += 1
        else:
            total_without += 1

    out_path = os.path.join(output_dir, f'{binary_name}_external.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {total_with} with ext calls, {total_without} without")


def save_vocabulary(vocab: dict, output_path: str):
    """Save external vocabulary."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({'vocab_size': len(vocab), 'vocabulary': vocab}, f, indent=2)
    print(f"  Vocabulary: {len(vocab)} tokens (including <NO_EXT>)")


def main():
    parser = argparse.ArgumentParser(description="Extract external function calls")
    parser.add_argument('--bir', help='BAP-IR file')
    parser.add_argument('--binary', help='Stripped binary (fallback to objdump)')
    parser.add_argument('--binary-name', required=True)
    parser.add_argument('--output-dir', default='data/external_calls')
    parser.add_argument('--vocab-path', default=None)
    args = parser.parse_args()

    if args.bir:
        print(f"Extracting external calls from {args.bir}...")
        calls = extract_from_bir(args.bir)
    else:
        raise ValueError("Provide --bir (BAP-IR file)")

    print(f"  Found {len(calls)} functions")
    save_external_calls(calls, args.binary_name, args.output_dir)

    # Build vocabulary
    vocab = build_vocabulary(calls, min_count=1)
    args.vocab_path and save_vocabulary(vocab, args.vocab_path)


if __name__ == '__main__':
    main()
