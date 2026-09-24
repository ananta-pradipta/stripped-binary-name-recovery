# Ghidra Jython post-script: decompile listed addresses, output ONE json per binary
# with per-address {ghidra_name, code}. Masking (name -> [MASK]) is done in
# post-processing on the host python (scripts/symgen_v2/build_symgen_input.py).
# args: <addr_list_file (0x hex per line)> <out_json> <elf_base_hex>
from ghidra.app.decompiler import DecompInterface
from ghidra.util.task import ConsoleTaskMonitor
import os, json

args = getScriptArgs()
addr_file, out_json, elf_base_hex = args[0], args[1], args[2]
delta = currentProgram.getImageBase().getOffset() - long(elf_base_hex, 16)
ifc = DecompInterface()
ifc.openProgram(currentProgram)
mon = ConsoleTaskMonitor()
fm = currentProgram.getFunctionManager()
res_map = {}
ok = fail = 0
for line in open(addr_file):
    a0 = line.strip()
    if not a0:
        continue
    a = long(a0, 16) + delta
    addr = toAddr(a)
    fn = fm.getFunctionAt(addr) or fm.getFunctionContaining(addr)
    if fn is None:
        try:
            fn = createFunction(addr, None)
        except:
            fn = None
    if fn is None:
        res_map[a0] = {'error': 'no_function'}; fail += 1
        continue
    try:
        r = ifc.decompileFunction(fn, 90, mon)
        if r is not None and r.decompileCompleted():
            res_map[a0] = {'ghidra_name': fn.getName(), 'code': r.getDecompiledFunction().getC()}
            ok += 1
        else:
            res_map[a0] = {'error': 'decomp_failed'}; fail += 1
    except Exception as e:
        res_map[a0] = {'error': 'exc'}; fail += 1
f = open(out_json, 'w')
json.dump(res_map, f)
f.close()
print('EXPORT_DONE ok=%d fail=%d' % (ok, fail))
