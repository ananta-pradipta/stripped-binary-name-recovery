"""Package-level paired bootstrap CIs and NCT-threshold sensitivity from per_pkg_dump.json."""
import json, sys, numpy as np
d=json.load(open(sys.argv[1])); rng=np.random.default_rng(0)
pk=[p for p,v in d.items() if v.get('SymGen') is not None]
print('packages with all systems:', len(pk))
def arr(k): return np.array([d[p][k] for p in pk])
routed=arr('routed_common'); G=arr('G_common'); R=arr('R_common'); SG=arr('SymGen'); BL=arr('BLens')
def ci(diff, B=10000):
    n=len(diff); idx=rng.integers(0,n,(B,n)); m=diff[idx].mean(1); return diff.mean(), np.percentile(m,2.5), np.percentile(m,97.5), (m<=0).mean()
for name,a,b in [('routed - SymGen',routed,SG),('routed - BLens',routed,BL),('routed - G',routed,G),('routed - R',routed,R),('G - SymGen',G,SG)]:
    m,lo,hi,p=ci(a-b); print('%-16s mean %.3f  95%% CI [%.3f, %.3f]  P(<=0)=%.4f' % (name,m,lo,hi,p))
print('pkg-level means: routed %.3f G %.3f R %.3f SymGen %.3f BLens %.3f' % (routed.mean(),G.mean(),R.mean(),SG.mean(),BL.mean()))
# NCT threshold sensitivity: family-linked packages stay NCT; others by seen_share threshold
allp=list(d.keys())
fam_linked=[p for p in allp if d[p]['regime']=='NCT' and d[p]['seen_share']<0.60]
print('family-linked NCT packages (seen share <60%):', fam_linked)
for thr in (0.5,0.6,0.7):
    nct=[p for p in allp if p in fam_linked or d[p]['seen_share']>=thr]; ft=[p for p in allp if p not in nct]
    def fn_mean(ps,k):
        n=sum(d[p]['n'] for p in ps); return sum(d[p][k]*d[p]['n'] for p in ps)/n
    print('thr %.2f: NCT %2d pkgs FT %2d pkgs | FT fn-F1 routed %.3f G %.3f R %.3f | NCT routed %.3f G %.3f R %.3f | FT pkg-F1 routed %.3f G %.3f R %.3f' % (thr,len(nct),len(ft),fn_mean(ft,'routed'),fn_mean(ft,'G'),fn_mean(ft,'R'),fn_mean(nct,'routed'),fn_mean(nct,'G'),fn_mean(nct,'R'),np.mean([d[p]['routed'] for p in ft]),np.mean([d[p]['G'] for p in ft]),np.mean([d[p]['R'] for p in ft])))
