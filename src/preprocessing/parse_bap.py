"""
Parse BAP-IR into per-function CFG graphs.

IMPROVED: Uses instruction-type tokenization instead of raw BAP-IR tokens.
Raw tokens (RSP, :=, mem, u64) → instruction types (STACK_OP, CALL_malloc, MEM_WRITE)

This reduces token vocabulary from 32,000+ to ~120 meaningful types.
"""
import json
import re
import os
import argparse
from typing import Dict, List


# ── Instruction classification ──

CALL_EXT_RE = re.compile(r'call\s+@(\w+)(?::external)?')
CALL_INT_RE = re.compile(r'call\s+@(sub_[0-9a-fA-F]+)')
CALL_INDIRECT_RE = re.compile(r'call\s+(?:mem|#\d|R[A-Z])')

STACK_REGS = {'RSP', 'RBP', 'ESP', 'EBP'}
ARG_REGS = {'RDI', 'RSI', 'RDX', 'RCX', 'R8', 'R9'}
RET_REGS = {'RAX', 'EAX'}
FLAGS = {'CF', 'OF', 'ZF', 'SF', 'PF', 'AF'}


def _get_imm_bucket(line: str) -> str:
    """Classify immediate values into semantic buckets."""
    # Find hex or decimal constants in the line (not register temps like #12345)
    for m in re.finditer(r'(?<!\#)\b0x([0-9a-fA-F]+)\b', line):
        val = int(m.group(1), 16)
        if val == 0:
            return '_ZERO'
        elif val == 1:
            return '_ONE'
        elif val <= 15:
            return '_SMALL'
        elif val & (val - 1) == 0:  # power of 2
            return '_POW2'
        elif val <= 255:
            return '_BYTE'
        elif val <= 0xFFFF:
            return '_WORD'
        elif val >= 0x10000:
            return '_ADDR'  # likely an address/string reference
        return ''
    # Check decimal constants
    for m in re.finditer(r'(?<!\#)(?<!\w)\b(\d+)\b(?![\]:}])', line):
        val = int(m.group(1))
        if val == 0:
            return '_ZERO'
        elif val == 1:
            return '_ONE'
        elif val <= 15:
            return '_SMALL'
        elif val & (val - 1) == 0:
            return '_POW2'
        elif val <= 255:
            return '_BYTE'
        return ''
    return ''


def _get_operand_size(line: str) -> str:
    """Extract operand size suffix from BAP-IR line."""
    m = re.search(r':u(\d+)', line)
    if m:
        return '_' + m.group(1)
    return ''


def _get_arith_op(line: str) -> str:
    """Classify arithmetic operation type."""
    # Check specific operators (order matters — check multi-char first)
    if '<<' in line or '>>' in line:
        return '_SHIFT'
    if ' ^ ' in line:
        return '_XOR'
    if ' | ' in line:
        return '_OR'
    if ' & ' in line:
        return '_AND'
    if ' * ' in line:
        return '_MUL'
    if ' / ' in line or ' %' in line:
        return '_DIV'
    if ' - ' in line:
        return '_SUB'
    if ' + ' in line:
        return '_ADD'
    return ''


def classify_instruction(raw_line: str) -> List[str]:
    """Classify a BAP-IR instruction into meaningful type tokens.

    V2: Richer tokenization — decomposes FLAG_SET into per-flag tokens,
    adds operand size suffixes, arithmetic operation types, and register
    class information. Grows vocab from ~370 to ~1500+ for better
    discrimination of 0-ext-call functions.
    """
    line = re.sub(r'^[0-9a-fA-F]+:\s*', '', raw_line).strip()
    if not line:
        return []

    # External call (MOST IMPORTANT — keep as-is)
    ext_match = CALL_EXT_RE.search(line)
    if ext_match:
        call_name = ext_match.group(1)
        if not call_name.startswith('sub_') and hextester(call_name.split('sub_')):
            return [f'CALL_{call_name}']
        else:
            # Count arg setup registers used before this call
            return ['CALL_INTERNAL']

    if CALL_INT_RE.search(line):
        return ['CALL_INTERNAL']
    if CALL_INDIRECT_RE.search(line):
        return ['CALL_INDIRECT']

    if 'noreturn' in line or (line.startswith('call') and 'return' in line):
        pass
    if re.match(r'.*\breturn\b', line) and 'call' not in line:
        return ['RETURN']

    if 'when' in line:
        # Enrich: what flag is being tested?
        for flag in FLAGS:
            if flag in line:
                return [f'COND_BRANCH_{flag}']
        return ['COND_BRANCH']
    if 'goto' in line:
        return ['BRANCH']

    # Per-flag decomposition (was: all → FLAG_SET)
    for flag in FLAGS:
        if line.startswith(flag + ' ') or line.startswith(flag + ':'):
            return [f'FLAG_{flag}']

    # Stack operations with size
    if any(reg in line for reg in STACK_REGS):
        sz = _get_operand_size(line)
        if ':=' in line and 'mem' in line and '<-' not in line:
            return [f'STACK_LOAD{sz}']
        elif 'mem' in line and '<-' in line:
            return [f'STACK_STORE{sz}']
        elif ':=' in line:
            return ['STACK_OP']
        return ['STACK_ACCESS']

    # Memory operations with size and immediate bucket
    sz = _get_operand_size(line)
    imm = _get_imm_bucket(line)
    if 'mem' in line and '<-' in line:
        if any(reg in line for reg in ARG_REGS):
            return [f'MEM_WRITE_ARG{sz}']
        elif any(reg in line for reg in RET_REGS):
            return [f'MEM_WRITE_RET{sz}']
        # Constant store (e.g., pushing address/string ref onto stack)
        if imm:
            return [f'MEM_WRITE_IMM{imm}{sz}']
        return [f'MEM_WRITE{sz}']
    if 'mem' in line and ':=' in line:
        if any(reg in line for reg in ARG_REGS):
            return [f'MEM_READ_ARG{sz}']
        elif any(reg in line for reg in RET_REGS):
            return [f'MEM_READ_RET{sz}']
        # Load from constant address (likely global/string)
        if imm == '_ADDR':
            return [f'MEM_READ_GLOBAL{sz}']
        return [f'MEM_READ{sz}']

    # Register-specific operations
    if ':=' in line and any(reg in line for reg in ARG_REGS):
        op = _get_arith_op(line)
        if op:
            return [f'ARG_ARITH{op}']
        # Distinguish: loading constant addr into arg reg (likely string/func ptr)
        if imm == '_ADDR':
            return ['ARG_LOAD_ADDR']
        elif imm:
            return [f'ARG_SETUP{imm}']
        return ['ARG_SETUP']
    if ':=' in line and any(reg in line for reg in RET_REGS):
        op = _get_arith_op(line)
        if op:
            return [f'RET_ARITH{op}']
        if imm == '_ADDR':
            return ['RET_LOAD_ADDR']
        return ['RETVAL']

    # Arithmetic with operation type
    if ':=' in line and any(op in line for op in ['+', '-', '*', '/', '^', '<<', '>>', '|', '&']):
        op = _get_arith_op(line)
        return [f'ARITH{op}']

    if any(op in line for op in ['<', '>']):
        return ['COMPARE']
    if ':=' in line:
        sz = _get_operand_size(line)
        if imm:
            return [f'ASSIGN{imm}{sz}'] if sz else [f'ASSIGN{imm}']
        return [f'ASSIGN{sz}'] if sz else ['ASSIGN']

    return ['OTHER']


# ── Skip patterns ──
SKIP_PATTERNS = [
    r'^\.plt$', r'^\.init$', r'^\.fini$', r'^\.text$',
    r'^\._', r'^_start$', r'^_init$', r'^_fini$',
    r'^register_tm_clones$', r'^deregister_tm_clones$',
    r'^frame_dummy$', r'^__do_global_',
    r'^__libc_csu_', r'^_dl_',
]
SKIP_RE = [re.compile(p) for p in SKIP_PATTERNS]


def hextester(s):
    if len(s) < 2:
        return False
    try:
        int(s[1], 16)
        return True
    except:
        return False

def should_skip_function(name):
    if not name:
        return True
    if name.startswith('sub_') and hextester(name.split('sub_')):
        return False
    return any(r.search(name) for r in SKIP_RE)


def get_function_address(func_name, line_hex):
    if func_name.startswith('sub_') and hextester(func_name.split('sub_')):
        return '0x' + func_name[4:]
    return '0x' + line_hex


def parse_bir_file(bir_path: str) -> Dict[str, dict]:
    """Parse BAP-IR with instruction-type tokenization."""
    functions = {}
    current_func = None
    current_func_name = None
    current_func_line_hex = None
    current_first_block_hex = None
    current_block_id = None
    current_block_label = None
    current_tokens = []
    current_ext_call = None
    current_internal_callees = []
    block_list = []
    edge_list = []
    block_id_map = {}
    block_counter = 0

    func_pattern = re.compile(r'^([0-9a-fA-F]+):\s+sub\s+(\S+?)\(')
    param_pattern = re.compile(r'^([0-9a-fA-F]+):.*::\s+(in|out)\s+')
    block_label_pattern = re.compile(r'^([0-9a-fA-F]+):\s*$')
    instruction_pattern = re.compile(r'^([0-9a-fA-F]+):\s+(.+)$')
    branch_pattern = re.compile(r'(?:return|goto)\s+%?([0-9a-fA-F]+)')
    program_pattern = re.compile(r'^[0-9a-fA-F]+:\s+program\s*$')

    def save_current_block():
        nonlocal current_block_id, current_tokens, current_ext_call
        if current_block_id is not None and current_tokens:
            block_list.append({
                'id': current_block_id,
                'label': current_block_label or '',
                'tokens': current_tokens,
                'num_tokens': len(current_tokens),
                'has_external_call': current_ext_call is not None,
                'external_call_name': current_ext_call,
            })
        current_tokens = []
        current_ext_call = None

    def save_current_function():
        nonlocal current_func, current_func_name, block_list, edge_list, block_id_map
        nonlocal current_first_block_hex, current_internal_callees
        save_current_block()
        if current_func and block_list:
            resolved_edges = []
            for src, dst_hex in edge_list:
                if dst_hex in block_id_map:
                    dst = block_id_map[dst_hex]
                    if [src, dst] not in resolved_edges:
                        resolved_edges.append([src, dst])
            for i in range(len(block_list) - 1):
                edge = [block_list[i]['id'], block_list[i + 1]['id']]
                if edge not in resolved_edges:
                    resolved_edges.append(edge)

            address = get_function_address(current_func_name, current_func)
            if not current_func_name.startswith('sub_') and hextester(current_func_name.split('sub_')) and current_first_block_hex:
                address = '0x' + current_first_block_hex

            functions[current_func_name] = {
                'address': address,
                'blocks': block_list,
                'edges': resolved_edges,
                'num_blocks': len(block_list),
                'num_edges': len(resolved_edges),
                'internal_callees': sorted(set(current_internal_callees)),
            }
        block_list = []
        edge_list = []
        block_id_map = {}
        current_first_block_hex = None
        current_internal_callees = []

    with open(bir_path, 'r') as f:
        for line_raw in f:
            line = line_raw.rstrip()
            if not line.strip():
                continue
            if program_pattern.match(line):
                continue

            func_match = func_pattern.match(line)
            if func_match:
                save_current_function()
                current_func = func_match.group(1)
                current_func_name = func_match.group(2)
                current_func_line_hex = func_match.group(1)
                current_first_block_hex = None
                current_block_id = None
                current_block_label = None
                block_counter = 0
                continue

            if param_pattern.match(line):
                continue

            block_match = block_label_pattern.match(line)
            if block_match and current_func:
                save_current_block()
                hex_label = block_match.group(1)
                current_block_id = block_counter
                current_block_label = hex_label
                block_id_map[hex_label] = block_counter
                if current_first_block_hex is None:
                    current_first_block_hex = hex_label
                block_counter += 1
                continue

            instr_match = instruction_pattern.match(line)
            if instr_match and current_func and current_block_id is not None:
                instr_text = instr_match.group(2).strip()

                # IMPROVED: Use instruction-type tokens instead of raw tokens
                type_tokens = classify_instruction(line)
                if type_tokens:
                    current_tokens.extend(type_tokens)

                # Track external calls
                ext_match = CALL_EXT_RE.search(instr_text)
                if ext_match:
                    call_name = ext_match.group(1)
                    if not call_name.startswith('sub_') and hextester(call_name.split('sub_')):
                        current_ext_call = call_name

                # Track internal callees
                int_match = CALL_INT_RE.search(instr_text)
                if int_match:
                    callee_name = int_match.group(1)
                    current_internal_callees.append(callee_name)

                # Track edges
                branch_matches = branch_pattern.findall(instr_text)
                for target_hex in branch_matches:
                    if current_block_id is not None:
                        edge_list.append((current_block_id, target_hex))

                continue

    save_current_function()
    return functions


def save_function_graphs(functions, binary_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    saved = 0
    skipped = 0
    for func_name, func_data in functions.items():
        if func_data['num_blocks'] < 1:
            skipped += 1
            continue
        if should_skip_function(func_name):
            skipped += 1
            continue
        output = {
            'binary': binary_name,
            'function_name': func_name,
            **func_data,
        }
        safe_name = re.sub(r'[^\w\-.]', '_', func_name)
        out_path = os.path.join(output_dir, f'{binary_name}_{safe_name}.json')
        with open(out_path, 'w') as f:
            json.dump(output, f, indent=2)
        saved += 1
    print(f"  Saved {saved} functions, skipped {skipped} (stubs/empty)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bir', required=True)
    parser.add_argument('--binary-name', required=True)
    parser.add_argument('--output-dir', default='data/graphs')
    args = parser.parse_args()

    print(f"  Parsing {args.bir}...")
    functions = parse_bir_file(args.bir)
    print(f"  Found {len(functions)} functions total")

    if not functions:
        print("  ⚠ No functions found!")
        return

    sample = list(functions.items())[:3]
    for name, data in sample:
        print(f"    {name:30s} @ {data['address']:12s} ({data['num_blocks']} blocks, tokens: {data['blocks'][0]['tokens'][:5]})")

    save_function_graphs(functions, args.binary_name, args.output_dir)


if __name__ == '__main__':
    main()
