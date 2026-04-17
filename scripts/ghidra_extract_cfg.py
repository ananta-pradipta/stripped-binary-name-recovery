# Ghidra headless script: Extract CFG + instruction tokens per function
# Run via: analyzeHeadless /tmp/ghidra_proj proj -import binary -postScript ghidra_extract_cfg.py output_dir
#
# Outputs one JSON per function in the same format as parse_bap.py
# @category BFNR

import json
import os
import sys
from ghidra.program.model.block import BasicBlockModel
from ghidra.program.model.listing import CodeUnit
from ghidra.util.task import ConsoleTaskMonitor

def get_v3_token(instr):
    """Map a Ghidra instruction to our V3 token vocabulary."""
    mnemonic = instr.getMnemonicString().upper()
    operands = instr.toString().upper()

    # Calls
    if mnemonic == 'CALL':
        op = instr.getDefaultOperandRepresentation(0)
        if op and not op.startswith('0x') and not op.startswith('['):
            # Named call (external or known function)
            name = op.split('.')[-1].strip()
            if name.startswith('_'):
                name = name.lstrip('_')
            return 'CALL_' + name
        elif '[' in operands:
            return 'CALL_INDIRECT'
        else:
            return 'CALL_INTERNAL'

    # Returns
    if mnemonic in ('RET', 'RETN'):
        return 'RETURN'

    # Conditional branches
    if mnemonic.startswith('J') and mnemonic != 'JMP':
        if 'Z' in mnemonic: return 'COND_BRANCH_ZF'
        if 'C' in mnemonic or 'B' in mnemonic: return 'COND_BRANCH_CF'
        if 'S' in mnemonic: return 'COND_BRANCH_SF'
        if 'O' in mnemonic: return 'COND_BRANCH_OF'
        if 'P' in mnemonic: return 'COND_BRANCH_PF'
        if 'L' in mnemonic or 'G' in mnemonic: return 'COND_BRANCH_ZF'
        return 'COND_BRANCH'

    # Unconditional jump
    if mnemonic == 'JMP':
        return 'BRANCH'

    # NOP
    if mnemonic in ('NOP', 'ENDBR64', 'ENDBR32'):
        return 'NOP'

    # Compare
    if mnemonic in ('CMP', 'TEST'):
        return 'COMPARE'

    # Stack operations
    if mnemonic == 'PUSH':
        return 'STACK_STORE'
    if mnemonic == 'POP':
        return 'STACK_LOAD_64'

    # MOV family
    if mnemonic.startswith('MOV'):
        if 'RSP' in operands or 'RBP' in operands:
            return 'STACK_OP'
        if 'RDI' in operands or 'RSI' in operands or 'RDX' in operands or 'RCX' in operands:
            if 'MOV' == mnemonic:
                return 'ARG_SETUP'
        if '[' in operands:
            # Memory access
            if operands.index('[') < operands.index(',') if ',' in operands else True:
                return 'MEM_READ_64'
            else:
                return 'MEM_WRITE_64'
        return 'ASSIGN'

    # LEA
    if mnemonic == 'LEA':
        if 'RDI' in operands or 'RSI' in operands:
            return 'ARG_LOAD_ADDR'
        return 'LOAD_ADDR'

    # Arithmetic
    if mnemonic in ('ADD', 'ADC'):
        if 'RSP' in operands: return 'STACK_OP'
        if 'RDI' in operands or 'RSI' in operands: return 'ARG_ARITH_ADD'
        return 'ARITH_ADD'
    if mnemonic in ('SUB', 'SBB'):
        if 'RSP' in operands: return 'STACK_OP'
        if 'RDI' in operands or 'RSI' in operands: return 'ARG_ARITH_SUB'
        return 'ARITH_SUB'
    if mnemonic in ('IMUL', 'MUL'):
        return 'ARITH_MUL'
    if mnemonic in ('IDIV', 'DIV'):
        return 'ARITH_DIV'
    if mnemonic in ('INC', 'DEC', 'NEG', 'NOT'):
        return 'ARITH_UNARY'

    # Shift/rotate
    if mnemonic in ('SHL', 'SHR', 'SAR', 'SAL', 'ROL', 'ROR'):
        return 'ARITH_SHIFT'

    # Bitwise
    if mnemonic in ('AND', 'OR', 'XOR'):
        if mnemonic == 'XOR' and len(set(operands.split(','))) == 1:
            return 'ASSIGN_ZERO'
        return 'ARITH_' + mnemonic

    # String operations
    if mnemonic in ('REP', 'REPZ', 'REPNZ', 'MOVSB', 'MOVSQ', 'STOSB', 'STOSQ', 'CMPSB'):
        return 'STRING_OP'

    # SIMD/SSE
    if mnemonic.startswith(('MOVAPS', 'MOVUPS', 'MOVDQA', 'MOVSS', 'MOVSD',
                            'ADDSS', 'ADDSD', 'MULSS', 'MULSD', 'SUBSS',
                            'PXOR', 'XORPS', 'XORPD', 'CVTS')):
        return 'SIMD_OP'

    # Flag setting
    if mnemonic in ('STC', 'CLC', 'STD', 'CLD', 'LAHF', 'SAHF'):
        return 'FLAG_OP'

    # CMOV
    if mnemonic.startswith('CMOV'):
        return 'CMOV'

    # SET
    if mnemonic.startswith('SET'):
        return 'SET_FLAG'

    # XCHG
    if mnemonic == 'XCHG':
        return 'XCHG'

    # CDQ/CQO (sign extend)
    if mnemonic in ('CDQ', 'CQO', 'CDQE', 'CBW', 'CWD', 'CWDE'):
        return 'SIGN_EXTEND'

    return 'OTHER'


def extract_cfg(program, func, binary_name):
    """Extract CFG for a single function."""
    monitor = ConsoleTaskMonitor()
    bbModel = BasicBlockModel(program)

    blocks = []
    edges = []
    block_map = {}  # address -> block_id
    callees = set()

    # Get all basic blocks
    block_iter = bbModel.getCodeBlocksContaining(func.getBody(), monitor)
    block_id = 0

    while block_iter.hasNext():
        bb = block_iter.next()
        addr = bb.getFirstStartAddress()
        block_map[addr.toString()] = block_id

        tokens = []
        has_ext_call = False
        ext_call_name = None

        # Iterate instructions in block
        listing = program.getListing()
        instr_iter = listing.getInstructions(bb, True)
        while instr_iter.hasNext():
            instr = instr_iter.next()
            token = get_v3_token(instr)
            tokens.append(token)

            # Track external calls
            if token.startswith('CALL_') and token not in ('CALL_INTERNAL', 'CALL_INDIRECT'):
                has_ext_call = True
                ext_call_name = token.replace('CALL_', '')

            # Track internal callees
            if token == 'CALL_INTERNAL':
                refs = instr.getReferencesFrom()
                for ref in refs:
                    if ref.getReferenceType().isCall():
                        target = ref.getToAddress()
                        target_func = program.getFunctionManager().getFunctionAt(target)
                        if target_func:
                            callees.add('sub_' + target.toString().lstrip('0'))

        blocks.append({
            'id': block_id,
            'label': addr.toString().lstrip('0') or '0',
            'tokens': tokens,
            'num_tokens': len(tokens),
            'has_external_call': has_ext_call,
            'external_call_name': ext_call_name,
        })
        block_id += 1

    # Get edges
    block_iter = bbModel.getCodeBlocksContaining(func.getBody(), monitor)
    while block_iter.hasNext():
        bb = block_iter.next()
        src_addr = bb.getFirstStartAddress().toString()
        src_id = block_map.get(src_addr)
        if src_id is None:
            continue

        dest_iter = bb.getDestinations(monitor)
        while dest_iter.hasNext():
            dest_ref = dest_iter.next()
            dest_addr = dest_ref.getDestinationAddress().toString()
            dest_id = block_map.get(dest_addr)
            if dest_id is not None and dest_id != src_id:
                edges.append([src_id, dest_id])

    func_addr = func.getEntryPoint().toString().lstrip('0') or '0'
    func_name = 'sub_' + func_addr

    return {
        'binary': binary_name,
        'function_name': func_name,
        'address': '0x' + func_addr,
        'blocks': blocks,
        'edges': edges,
        'num_blocks': len(blocks),
        'num_edges': len(edges),
        'internal_callees': list(callees),
    }


# Main script
args = getScriptArgs()
if len(args) < 2:
    print("Usage: ghidra_extract_cfg.py <binary_name> <output_dir>")
    sys.exit(1)

binary_name = args[0]
output_dir = args[1]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

program = currentProgram
func_manager = program.getFunctionManager()
func_iter = func_manager.getFunctions(True)

count = 0
while func_iter.hasNext():
    func = func_iter.next()
    # Skip thunks and external functions
    if func.isThunk() or func.isExternal():
        continue
    # Skip very small functions (likely stubs)
    if func.getBody().getNumAddresses() < 2:
        continue

    cfg = extract_cfg(program, func, binary_name)

    if cfg['num_blocks'] > 0:
        out_path = os.path.join(output_dir, cfg['function_name'] + '.json')
        # Use full binary_function naming
        out_name = binary_name + '_' + cfg['function_name']
        out_path = os.path.join(output_dir, out_name + '.json')
        with open(out_path, 'w') as f:
            json.dump(cfg, f)
        count += 1

print("Extracted {} functions from {}".format(count, binary_name))
