# P1v2 Phase Report — Enriched-Input Composition Heads (V3e / lex / copy)

**Dates:** 2026-08-14 → 2026-08-16 · **Branch:** `unified` · **Protocol:** D_train = 241,109 paper-clean functions (production ids), PN-dev = 2,000 held-out complete names (G1-identical), single-shot clean-7 eval (13,084/13,581 protocol fns; subset-matched), fold-0 package-holdout retention. Gates: clean-7 NOVEL ≥ 0.12 and ≥ U1+0.03; retention ≥ 0.6. Retrieval pipeline untouched (user constraint); all inputs from separate enriched dirs (`data/graphs_v2`, `data/lex_v1`, caches `results/p1v2`, `results/p1lex`).

## Architecture (shared skeleton)

Per function: ≤128 CFG blocks × ≤24 token ids → frozen CE trunk (production
`best_model_cont_control.pt` block encoder: 256-d embedding + 4-layer
transformer; base rows 0–2999 frozen byte-identical, rows ≥3000 trainable via
row-masked grads, emb param group wd=0) → mean-pool per block → MLP adapter
(512→256) → token-conditioned attention head scoring V=5,400 name sub-tokens
(multi-label). **No GAT** (random-init GATConv homogenized blocks: pairwise
cos 0.02–0.33 → 0.25–0.60; removal turned a declining curve into the first
climbing one). No adversarial term. BCE with pos_weight 5, G1 hard negatives,
batch 64, frozen trunk throughout, a100_20g MIG lane.

## Variants and results

| variant | input | PN peak | clean-7 ALL | NOVEL | PARTIAL-OOV | AUPRC | retention | go |
|---|---|---|---|---|---|---|---|---|
| control | V3e tokens, enrichment rows FROZEN at random init (≈V3-only signal) | 0.0686 (ep8) | 0.1184 | 0.0297 | 0.0153 | 0.0597 | 0.232 | ✗ |
| embfix | + enrichment rows trained (LIT_*/GREF literals) | 0.0727 (ep4) | — (superseded) | — | — | — | — | — |
| P1-lex | + lexical pseudo-block (LEXS_* string words, LEXE_* ext names; 43% train / 49% clean-7 coverage) | 0.0745 (ep8) | 0.1144 | 0.0324 | 0.0188 | 0.0736 | 0.212 | ✗ |
| **P1-copy** | + direct copy wire: `score[t] += copyw·1[t ∈ lex_words(fn)]` | **0.0795 (ep8)** | **0.1247** | **0.0385** | **0.0346** | **0.0830** | **0.317** | ✗ |

Baselines: U1 NOVEL 0.0894 · E1-B 0.0693 · E2-C 0.0518 (25M production
encoder states; not capacity-comparable to this 4-layer trunk).

## Zero-training counterpart (retrieval side, banked)

`experiments_semantic/u1_lex_union.py` (commit 5f343abc): when U0∩U1 = ∅
(test-time signal of lost retrieval), union df<0.2-filtered lexical atoms
into U1's prediction. Single-shot clean-7, gate fixed a priori:
**NOVEL 0.0931 → 0.1008 (+0.0077, CI [+0.0047,+0.0107]); ALL +0.0036
(CI [+0.0028,+0.0043]); SEEN no cost.** Strictly dominant. First real
cross-project NOVEL gain over U1 in the research arc.

## Findings

1. **Evidence path length is the controlling variable.** PN peaks are
   monotone in routing directness: 0.0686 (none) → 0.0727 (trunk-fed
   literals) → 0.0745 (trunk-fed words) → 0.0795 (direct wire). GT
   diagnosis showed trunk-fed lexical evidence is destroyed before the head
   (function with string "bfd" doesn't predict `bfd`; P1-lex ≈ embfix
   paired 45:45 on identical functions).
2. **The lexical channel is worth ≈ +0.01 NOVEL cross-project regardless of
   routing.** Learned wire: +0.0088 over control; zero-training U1-union:
   +0.0077 over U1. Both ≈ half the measured perfect-exploitation ceiling
   (+0.02: only 47% of novel fns have evidence; it covers ~10% of their
   name tokens at 6.7% precision). The binding constraint is evidence
   coverage, not modeling.
3. **Copy-wire signal is more portable than trunk signal.** Retention 0.317
   vs 0.212–0.232; PN→clean-7 shrink 0.48× vs 0.43×. Directionally clean,
   but most head knowledge remains package-local.
4. **Head family closed.** All variants fail both gates by wide margins
   (best NOVEL 0.0385 vs U1 0.0894 vs gate 0.12). With a frozen small trunk,
   no input enrichment closes the gap; with trunk fine-tuning, training
   collapses (5 prior failures). Composition-by-learned-head is dominated
   by retrieval-composition plus direct evidence rules.

## Mechanical post-mortems banked this phase

- Random-init GATConv on frozen states = block homogenizer (do not reuse).
- `P1_FREEZE` froze the whole embedding table → enrichment rows were
  N(0,1) noise for the first run (verify EFFECT, not config).
- AdamW decoupled weight decay moves zero-grad params → frozen rows decayed
  until emb got its own wd=0 group (caught by effect-preflight).
- PN→clean-7 shrink 0.43–0.48×; PN is selection-only.

## Artifacts

- Code: `experiments_semantic/{parse_bap_v2 → src/preprocessing/parse_bap_v2.py, p1_cache_v2.py, p1_cache_v3.py, p1_lex_extract.py, p1v2_train.py, u1_lex_union.py}`
- Data: `data/graphs_v2` (395K enriched graphs), `data/lex_v1` (834 binaries), caches `results/p1v2`, `results/p1lex`
- Checkpoints (Wulver `strlex_ws/results/p1v2/`): `p1_ckpt_{P1-full,P1-embfix,P1-lex,P1-copy}.pt` (+fold0), summaries `p1_summary_{tag}.json`
- Key commits: 2b51f999 (NOGAT/NOADV), 6f4db119 (EMB_TRAIN+TAG), 1c6de942 (wd=0), 6af5ab9f (lex pipeline), 5f343abc (U1-union rule), 19d2743b-era (P1_COPY)

## Recommended next directions (direct-evidence spirit)

1. Refine the U1-union rule: cross-binary df junk filter; string>ext source
   weighting; dev-tuned partial-disagreement trigger (~+0.010 headroom in
   the transparent sweep).
2. Expand evidence coverage (the binding constraint): rodata linkage beyond
   direct constants (GREF address chasing through .data pointers), caller/
   callee lexical inheritance (a wrapper inherits its callee's strings).
3. If a learned component returns: the copy wire pattern (path-length-1,
   per-source weights) — never trunk-fed evidence.
