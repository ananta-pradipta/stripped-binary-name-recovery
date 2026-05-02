import json, glob, os
matched = 0
unmatched = 0
match_index = {}

total = (len(glob.glob('data/labels/*_labels.json')))
curr = 0

# ── Load labels: build int_address → name mapping per binary ──
labels_by_binary = {}
for lf in sorted(glob.glob('data/labels/*_labels.json')):
    with open(lf) as f:
        data = json.load(f)
    binary = data['binary']

    # Build mapping: integer address → function name
    int_to_name = {}
    funcs = data.get('functions', {})
    addr_to_name = data.get('addr_to_name', {})

    print(f"{100*curr/total:.2f}% Complete")

    # Get matching graph
    for function in addr_to_name:
        faddr = int(function, 16)
        faddrAfterBR64 = faddr+4 # BAP skips endbr64, sound because functions are 16 byte aligned
        if(os.path.isfile(f"data/graphs/{binary}_sub_{hex(faddr)[2:]}.json")):
            fname = f"data/graphs/{binary}_sub_{hex(faddr)[2:]}.json"
        elif os.path.isfile(f"data/graphs/{binary}_sub_{hex(faddrAfterBR64)[2:]}.json"):
            fname = f"data/graphs/{binary}_sub_{hex(faddrAfterBR64)[2:]}.json"
        elif os.path.isfile(f"data/graphs/{binary}_{addr_to_name[function]}.json"):
            fname = f"data/graphs/{binary}_{addr_to_name[function]}.json"
        else:
            unmatched += 1
            continue
        with open(fname, "r") as f:
            graph = json.load(f)

            address_str = graph.get('address', '')
            func_name = graph.get('function_name', '')
            match_index[fname] = {
                'binary': binary,
                'address': address_str,
                'address_int': faddr,
                'bap_name': func_name,
                'real_name': addr_to_name[function],
            }
            matched += 1
    curr += 1

# Save
with open('data/match_index.json', 'w') as f:
    json.dump(match_index, f, indent=2)

print(f"\n  ════════════════════════════════")
print(f"  Matched:   {matched}")
print(f"  Unmatched: {unmatched}")
print(f"  Rate:      {100*matched/max(matched+unmatched,1):.1f}%")
print(f"  ════════════════════════════════")

# Show sample matches
if match_index:
    print(f"\n  Sample matches:")
    for gf, info in list(match_index.items())[:10]:
        print(f"    {info['bap_name']:25s} → {info['real_name']:30s} @ {info['address']}")
else:
    # Debug: show what addresses exist in each
    print(f"\n  ⚠ ZERO matches. Debugging address formats:")
    if labels_by_binary:
        first_bin = next(iter(labels_by_binary))
        label_addrs = sorted(labels_by_binary[first_bin].keys())[:5]
        print(f"    Label addresses ({first_bin}): {[hex(a) for a in label_addrs]}")

    sub_graphs = [gf for gf in graph_files if 'sub_' in os.path.basename(gf)]
    if sub_graphs:
        with open(sub_graphs[0]) as f:
            g = json.load(f)
        print(f"    Graph address: {g['address']} = {int(g['address'], 16)}")


