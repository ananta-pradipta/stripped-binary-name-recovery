# Phase 3 — Evidence-Grounded Composition: Compact Negative-Result Table

The semantic information is often visibly present in stripped-binary evidence,
but no selector we built converts it into trustworthy **new** semantic concepts
under the frozen representation. H3 is dropped from the deployed architecture
and retained only as the evidence study / optional provenance analysis.

| stage | what was tried | key metric | result | verdict |
|---|---|---|---|---|
| **Evidence census** | count GT semantic atoms observable in permitted evidence (external calls, dynsym imports, strings, libraries; defined `.dynsym` excluded) | fraction of retrieval-F1=0 functions with ≥1 correct atom in evidence | **55.5%** available; residual-after-H1/H2 (REA) 54.0% of retr-fail fns | information is present |
| **H3-v1** | pointwise `z`-conditioned relevance MLP over evidence candidates | independent semantic rescue (retr-F1=0) | **0.0%** (re-finds only what H1/H2 already had) | STOP |
| **H3-v2** | residual-aware joint cross-attention selector (K=2 queries, residual-weighted loss, hard negatives) | H3-new-beyond-H1/H2 precision (leakage-free) | peaks **0.044**; no operating point ≥ 0.25; per-seed rescue (~0.19) does not survive ensembling (→0.019) | STOP |
| **Corroboration** | reuse H3-v2 scores to *verify* H1/H2 concepts (no new concepts) | P(correct \| grounded) − P(correct \| ungrounded) | lift real (**ALL +0.117, retr0 +0.196, novel +0.390**) but coverage **3.7%** and no ranking gain over H1/H2 confidence | DROP (keep as provenance badge only) |

**Scientific statement.** *Correct semantics are frequently observable in
lexical program evidence, but neither pointwise nor joint residual-aware
selectors could convert that evidence into trustworthy new semantic concepts
under the frozen function representation. Evidence grounding predicts concept
correctness, but too sparsely to change the deployed confidence ranking.*

**Implementation note.** An eval-time candidate-selection leak was found and
fixed in H3-v2 (`cand_tensors` prioritized GT positives when capping
candidates; used on the query/val sets). The leakage-free decomposed sweep is
the record; the fix does not change the verdict (the H3-new precision ceiling
of 0.044 is leakage-free and is what closes the branch).

**Do not claim** the 55.5% evidence-availability figure as achieved semantic
recovery; it establishes *availability ≠ trustworthy selectability*.
**Do not headline** the per-seed 19% H3 rescue; it is latent recall inside a
low-precision flood and does not survive ensemble aggregation.
