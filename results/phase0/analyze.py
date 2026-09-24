import json, os, re, collections, statistics, itertools
ROOT='$HOME/cs785-project'
OUTD='/tmp/claude-1000/-home-USER-cs785-project/0f684b34-c85e-456a-af39-b91d4f421f48/scratchpad/phase0_audit'
os.chdir(ROOT)
audit={}

# ---------- load scan ----------
mi=json.load(open('data/match_index.json'))
mi_bins=set(v['binary'] for v in mi.values())
lab_bins=set(f[:-12] for f in os.listdir('data/labels') if f.endswith('_labels.json'))
ext_bins=set(f[:-14] for f in os.listdir('data/external_calls') if f.endswith('_external.json'))
known_bins=mi_bins|lab_bins|ext_bins
by_len=sorted(known_bins,key=len,reverse=True)
prefix_index=collections.defaultdict(list)
for b in known_bins: prefix_index[b.split('_')[0]].append(b)
for k in prefix_index: prefix_index[k].sort(key=len,reverse=True)

def infer_binary(fn):
    stem=fn[:-5]
    pk=stem.split('_')[0]
    for b in prefix_index.get(pk,[]):
        if stem.startswith(b+'_'): return b, stem[len(b)+1:]
    if '_sub_' in stem:
        i=stem.index('_sub_'); return stem[:i], stem[i+1:]
    return None,None

rows={}  # fn -> dict
n_err=0; n_inferred=0; n_uninferred=0
with open(f'{OUTD}/scan.tsv') as f:
    for line in f:
        p=line.rstrip('\n').split('\t')
        fn,binary,fname,is_sub,has_ext,ntok,nblk,h,thunk,addr,callee0=p[:11]
        if len(p)>11 or ntok=='-1':
            n_err+=1; continue
        if binary=='' :
            b2,f2=infer_binary(fn)
            if b2 is None: n_uninferred+=1; continue
            binary,fname=b2,f2; n_inferred+=1
            is_sub='1' if fname.startswith('sub_') else '0'
        rows[fn]=dict(b=binary,f=fname,sub=is_sub=='1',ext=has_ext=='1',ntok=int(ntok),nblk=int(nblk),h=h,thunk=thunk=='1',addr=addr,callee0=callee0)
audit['scan']=dict(n_files=len(rows)+n_err+n_uninferred,n_parsed=len(rows),n_json_errors=n_err,
                   n_missing_binary_field_inferred_from_filename=n_inferred,n_unresolvable=n_uninferred,
                   n_sub_graphs=sum(1 for r in rows.values() if r['sub']),
                   n_nonsub_graphs=sum(1 for r in rows.values() if not r['sub']),
                   subsampled=False)
print(audit['scan'])

def pkg_of(b): return b.split('_')[0]
def opt_of(b):
    m=re.search(r'_(O[0-3s])$',b); return m.group(1) if m else 'default'
def bin_of(b):
    o=opt_of(b); return b[:-(len(o)+1)] if o!='default' else b

# ---------- resolve thunk hashes (loader-style) ----------
resolved_h={}
n_thunk=0; n_thunk_resolved=0
for fn,r in rows.items():
    if r['thunk']:
        n_thunk+=1
        cfn=f"{r['b']}_{r['callee0']}.json"
        if r['callee0'] and cfn in rows:
            resolved_h[fn]=rows[cfn]['h']; n_thunk_resolved+=1
        else: resolved_h[fn]=r['h']
    else: resolved_h[fn]=r['h']
audit['thunks']=dict(n_thunk_candidates_all_graphs=n_thunk,n_resolved_to_callee_graph=n_thunk_resolved)

# ================= B2 =================
per_bin=collections.defaultdict(lambda: dict(n_sub=0,n_sub_ext=0,n_matched=0,n_matched_ext=0))
mi_keys_by_bin=collections.defaultdict(list)
for k,v in mi.items(): mi_keys_by_bin[v['binary']].append(k)
matched_fns=set(os.path.basename(k) for k in mi)
for fn,r in rows.items():
    if not r['sub']: continue
    d=per_bin[r['b']]; d['n_sub']+=1; d['n_sub_ext']+=r['ext']
    if fn in matched_fns:
        d['n_matched']+=1; d['n_matched_ext']+=r['ext']
# external.json
extj={}
for b in ext_bins:
    d=json.load(open(f'data/external_calls/{b}_external.json'))
    fs=d.get('functions',[])
    n_any=sum(1 for x in fs if x.get('external_calls'))
    n_sub=sum(1 for x in fs if x.get('external_calls') and str(x.get('function_name','')).startswith('sub_'))
    extj[b]=dict(n_funcs=len(fs),n_with_calls=n_any,n_sub_with_calls=n_sub)
b2_bins={}
for b in sorted(set(per_bin)|set(extj)):
    d=per_bin.get(b,dict(n_sub=0,n_sub_ext=0,n_matched=0,n_matched_ext=0)); e=extj.get(b,dict(n_funcs=0,n_with_calls=0,n_sub_with_calls=0))
    frac=d['n_sub_ext']/d['n_sub'] if d['n_sub'] else None
    fracm=d['n_matched_ext']/d['n_matched'] if d['n_matched'] else None
    b2_bins[b]=dict(pkg=pkg_of(b),n_sub_graphs=d['n_sub'],n_sub_with_CALL_sym=d['n_sub_ext'],frac_sub_with_CALL_sym=frac,
                    n_matched=d['n_matched'],n_matched_with_CALL_sym=d['n_matched_ext'],frac_matched_with_CALL_sym=fracm,
                    extjson_n_funcs=e['n_funcs'],extjson_n_with_calls=e['n_with_calls'],extjson_n_sub_with_calls=e['n_sub_with_calls'],
                    has_extjson=b in extj)
flag=[b for b,x in b2_bins.items() if x['extjson_n_with_calls']>=10 and x['n_sub_graphs']>0 and x['n_sub_with_CALL_sym']==0]
b2_pkg=collections.defaultdict(lambda: dict(n_bins=0,n_sub=0,n_sub_ext=0,n_matched=0,n_matched_ext=0,n_flagged=0,extjson_with_calls=0))
for b,x in b2_bins.items():
    p=b2_pkg[x['pkg']]; p['n_bins']+=1; p['n_sub']+=x['n_sub_graphs']; p['n_sub_ext']+=x['n_sub_with_CALL_sym']
    p['n_matched']+=x['n_matched']; p['n_matched_ext']+=x['n_matched_with_CALL_sym']; p['n_flagged']+=(b in flag); p['extjson_with_calls']+=x['extjson_n_with_calls']
for p in b2_pkg.values():
    p['frac_sub']=p['n_sub_ext']/p['n_sub'] if p['n_sub'] else None
    p['frac_matched']=p['n_matched_ext']/p['n_matched'] if p['n_matched'] else None
tot_sub=sum(x['n_sub_graphs'] for x in b2_bins.values()); tot_ext=sum(x['n_sub_with_CALL_sym'] for x in b2_bins.values())
tot_m=sum(x['n_matched'] for x in b2_bins.values()); tot_me=sum(x['n_matched_with_CALL_sym'] for x in b2_bins.values())
audit['B2']=dict(overall_frac_sub_graphs_with_CALL_sym=tot_ext/tot_sub,overall_frac_matched_with_CALL_sym=tot_me/tot_m,
                 n_bins_flagged=len(flag),flagged_binaries=flag,
                 n_bins_zero_channel=sum(1 for x in b2_bins.values() if x['n_sub_graphs']>0 and x['n_sub_with_CALL_sym']==0),
                 n_bins_total=sum(1 for x in b2_bins.values() if x['n_sub_graphs']>0),
                 per_package={k:b2_pkg[k] for k in sorted(b2_pkg,key=lambda k:(b2_pkg[k]['frac_sub'] if b2_pkg[k]['frac_sub'] is not None else -1))},
                 per_binary=b2_bins)
print('B2 overall',audit['B2']['overall_frac_sub_graphs_with_CALL_sym'],'flagged',len(flag))

# ================= B3 =================
def load_labels(b):
    d=json.load(open(f'data/labels/{b}_labels.json'))
    if 'name_to_addr' in d: n2a=d['name_to_addr']
    else:
        F=d['functions']
        if F and next(iter(F)).startswith('0x'): n2a={v:k for k,v in F.items()}
        else: n2a=F
    return d.get('num_functions',len(n2a)), n2a
sub_by_bin=collections.defaultdict(list)
for fn,r in rows.items():
    if r['sub']: sub_by_bin[r['b']].append(fn)
b3={}
for b in sorted(lab_bins|mi_bins|set(sub_by_bin)):
    nl,n2a=load_labels(b) if b in lab_bins else (0,{})
    addrs=set(int(a,16) for a in n2a.values()) if n2a else set()
    nm=len(mi_keys_by_bin.get(b,[])); ng=len(sub_by_bin.get(b,[]))
    b3[b]=dict(pkg=pkg_of(b),has_labels=b in lab_bins,n_label_names=nl,n_label_unique_addrs=len(addrs),n_matched=nm,n_sub_graphs=ng,
               match_rate_by_addr=(nm/len(addrs) if addrs else None),match_rate_by_names=(nm/nl if nl else None))
low=[b for b,x in b3.items() if x['match_rate_by_addr'] is not None and x['match_rate_by_addr']<0.90]
zero=[b for b,x in b3.items() if x['has_labels'] and x['n_label_unique_addrs']>0 and x['n_sub_graphs']>0 and x['n_matched']==0]
nolab=[b for b,x in b3.items() if not x['has_labels'] and x['n_matched']>0]
graphs_no_mi=[b for b,x in b3.items() if x['n_sub_graphs']>0 and x['n_matched']==0]
# offset characterisation for 5 zero-match bins
offs={}
for b in zero[:5]:
    _,n2a=load_labels(b)
    gaddrs=sorted(int(rows[fn]['f'][4:],16) for fn in sub_by_bin[b])
    import bisect
    samp=[]
    for name,a in sorted(n2a.items())[:10]:
        ai=int(a,16); i=bisect.bisect_left(gaddrs,ai)
        cands=[gaddrs[j] for j in (i-1,i) if 0<=j<len(gaddrs)]
        near=min(cands,key=lambda g:abs(g-ai))
        samp.append(dict(name=name,label_addr=hex(ai),nearest_sub=hex(near),delta=near-ai))
    offs[b]=dict(n_sub_graphs=len(gaddrs),graph_addr_range=[hex(gaddrs[0]),hex(gaddrs[-1])],
                 label_addr_range=[hex(min(int(a,16) for a in n2a.values())),hex(max(int(a,16) for a in n2a.values()))],samples=samp)
rates=[x['match_rate_by_addr'] for x in b3.values() if x['match_rate_by_addr'] is not None]
audit['B3']=dict(n_bins_with_labels=len(lab_bins),n_bins_in_match_index=len(mi_bins),n_bins_with_sub_graphs=len(sub_by_bin),
                 total_label_unique_addrs=sum(x['n_label_unique_addrs'] for x in b3.values()),total_matched=len(mi),
                 total_sub_graphs=sum(x['n_sub_graphs'] for x in b3.values()),
                 median_match_rate=statistics.median(rates),mean_match_rate=statistics.mean(rates),
                 n_bins_rate_lt_090=len(low),bins_rate_lt_090={b:b3[b] for b in low},
                 n_bins_zero_match_with_labels_and_graphs=len(zero),bins_zero_match=zero,
                 bins_matched_but_no_labels_file=nolab,
                 n_bins_graphs_but_zero_matches=len(graphs_no_mi),
                 zero_match_offset_samples=offs,per_binary=b3)
print('B3 low',len(low),'zero',len(zero),'median',statistics.median(rates))

# ================= B5 =================
# real_name -> (hash, resolved hash, ntok) per binary via match_index
name_map=collections.defaultdict(dict)  # binary -> real_name -> fn
dup_names=collections.Counter()
for k in sorted(mi):
    v=mi[k]; fn=os.path.basename(k)
    if fn not in rows: continue
    if v['real_name'] in name_map[v['binary']]: dup_names[v['binary']]+=1; continue
    name_map[v['binary']][v['real_name']]=fn
groups=collections.defaultdict(dict)
for b in mi_bins: groups[bin_of(b)][opt_of(b)]=b
pairs=[]
for pb,opts in sorted(groups.items()):
    if len(opts)<2: continue
    for o1,o2 in itertools.combinations(sorted(opts),2):
        b1,b2_=opts[o1],opts[o2]
        n1,n2=name_map[b1],name_map[b2_]
        shared=set(n1)&set(n2)
        if not shared: 
            pairs.append(dict(pkg_bin=pb,opt1=o1,opt2=o2,bin1=b1,bin2=b2_,n_shared=0)); continue
        raw=sum(1 for n in shared if rows[n1[n]]['h']==rows[n2[n]]['h'])
        res=sum(1 for n in shared if resolved_h[n1[n]]==resolved_h[n2[n]])
        big=[n for n in shared if max(rows[n1[n]]['ntok'],rows[n2[n]]['ntok'])>=10]
        resb=sum(1 for n in big if resolved_h[n1[n]]==resolved_h[n2[n]])
        pairs.append(dict(pkg_bin=pb,pkg=pkg_of(b1),opt1=o1,opt2=o2,bin1=b1,bin2=b2_,n_shared=len(shared),
                          identity_raw=raw/len(shared),identity_thunk_resolved=res/len(shared),
                          n_shared_ge10tok=len(big),identity_resolved_ge10tok=(resb/len(big) if big else None)))
hi=[p for p in pairs if p['n_shared']>0 and p['identity_thunk_resolved']>=0.5]
def hist(vals):
    bins=[0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,1.0001]
    c=collections.OrderedDict()
    for lo,hi_ in zip(bins,bins[1:]):
        c[f'[{lo},{hi_ if hi_<=1 else 1.0}{")" if hi_<=1 else "]"}']=sum(1 for v in vals if lo<=v<hi_)
    return c
byopt=collections.defaultdict(list)
for p in pairs:
    if p['n_shared']>0: byopt[f"{p['opt1']}-{p['opt2']}"].append(p['identity_thunk_resolved'])
audit['B5']=dict(n_pkg_bin_with_ge2_opts=sum(1 for o in groups.values() if len(o)>=2),n_pairs=len(pairs),
                 n_pairs_with_shared=sum(1 for p in pairs if p['n_shared']>0),
                 n_pairs_identity_ge_0_5=len(hi),pairs_identity_ge_0_5=sorted(hi,key=lambda p:-p['identity_thunk_resolved']),
                 distribution_hist_thunk_resolved=hist([p['identity_thunk_resolved'] for p in pairs if p['n_shared']>0]),
                 distribution_hist_raw=hist([p['identity_raw'] for p in pairs if p['n_shared']>0]),
                 per_optpair_median={k:statistics.median(v) for k,v in sorted(byopt.items())},
                 per_optpair_n={k:len(v) for k,v in sorted(byopt.items())},
                 per_optpair_max={k:max(v) for k,v in sorted(byopt.items())},
                 n_binaries_with_duplicate_real_names=len(dup_names),n_duplicate_name_entries_skipped=sum(dup_names.values()),
                 all_pairs=pairs)
print('B5 pairs',len(pairs),'hi',len(hi))

# ================= B6 =================
nonsub_mi=[k for k in mi if '_sub_' not in os.path.basename(k)]
nonsub_names=collections.Counter(r['f'] for r in rows.values() if not r['sub'])
audit['B6']=dict(n_match_index_keys_non_sub=len(nonsub_mi),match_index_non_sub_keys=[dict(key=k,**mi[k]) for k in nonsub_mi],
                 n_non_sub_graph_files=sum(nonsub_names.values()),n_distinct_non_sub_names=len(nonsub_names),
                 top20_non_sub_names=nonsub_names.most_common(20),
                 loader_iterates_match_index_only=True,
                 loader_code=dict(unified='src/preprocessing/build_dataset.py:228 `for graph_path, match_info in self.match_index.items():` (+229 `if not os.path.exists(graph_path): continue`)',
                                  dev='git show dev:src/preprocessing/build_dataset.py:215 same loop'),
                 verdict='Loader never globs data/graphs; it opens exactly the match_index keys that exist on disk. So the 42 non-sub match_index entries ARE loaded (as regular samples with real_name label); the other non-sub graph files are never loaded (except indirectly: thunk resolution reads <binary>_<callee>.json for callee0 of 1-2-token CALL_INTERNAL graphs, and callee/caller signature lookup uses graphs already in match_index only).')

# ================= B8 =================
s=json.load(open('data/split_assignments.json'))
sets={k:set(s[k]) for k in ('train','val','test','excluded')}
known=set().union(*sets.values())
missing=sorted(mi_bins-known)
grp=collections.defaultdict(list)
for b in missing: grp[pkg_of(b)].append(b)
n_fn_missing=sum(len(mi_keys_by_bin[b]) for b in missing)
pk_check={}
for pk in ['dash','gettext','psmisc','recutils','nginx118','angie','tengine','grep','sed']:
    pk_check[pk]={k:sorted(b for b in sets[k] if pkg_of(b)==pk) for k in sets}
    pk_check[pk]['in_match_index']=sorted(b for b in mi_bins if pkg_of(b)==pk)
    pk_check[pk]['in_labels_dir']=sorted(b for b in lab_bins if pkg_of(b)==pk)
    pk_check[pk]['in_match_index_but_unsplit(default->train)']=sorted(b for b in missing if pkg_of(b)==pk)
audit['B8']=dict(split_sizes={k:len(v) for k,v in sets.items()},n_mi_bins=len(mi_bins),
                 n_mi_bins_not_in_split_file=len(missing),n_mi_functions_in_unsplit_bins=n_fn_missing,
                 frac_mi_functions_in_unsplit_bins=n_fn_missing/len(mi),
                 unsplit_bins_by_package={k:grp[k] for k in sorted(grp)},
                 n_unsplit_packages=len(grp),
                 split_bins_not_in_match_index=sorted(known-mi_bins),
                 package_split_membership=pk_check,
                 loader_behaviour='get_splits(): new_binaries = all_binaries - known; train_bins_set.update(new_binaries) (unified L671-673, dev L558-562)')
print('B8 missing',len(missing),n_fn_missing)

json.dump(audit,open(f'{OUTD}/audit.json','w'),indent=1)
print('wrote audit.json')
