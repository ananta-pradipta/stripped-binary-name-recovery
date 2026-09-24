# Unified Retrieval–Composition Research Arc — Consolidated Experiment Log
**Branch:** `unified` · **Protocol of record:** D_train = 241,174 paper-clean functions (243,289 − grep 1,267 − sed 848); clean-7 cross-project eval = 13,581 functions keyed `binary:address` (tengine, angie, nginx118, recutils, dash, gettext, psmisc); subset-matched comparisons where coverage < 100%. Compiled 2026-08-16.

---

## 0. Anchors and baselines

| system | metric | value |
|---|---|---|
| U0 (whole-name retrieval, rebuilt + anchored) | clean-7 macro-F1 / EM | **0.5855 / 48.6%** |
| U1 (retrieval token composition, seed 42 dump) | clean-7 ALL / NOVEL | **0.3199 / 0.0931** (multi-seed official NOVEL 0.0894) |
| Gates for any new composition head | clean-7 NOVEL | ≥ 0.12 and ≥ U1+0.03; retention ≥ 0.6 |
| PSEUDO-NOVEL dev (PN) | selection-only metric | 2,000 held-out complete names; PN→clean-7 shrink measured 0.43–0.48× |

---

## 1. Unified composition experiments (closed 2026-08-12, job 1174048)

| experiment | result | verdict |
|---|---|---|
| U0 rebuild + anchor | 0.5855 macro-F1 | anchor reproduced exactly |
| Add-only fusion (decoder over retrieval) | **−0.0336 ALL; 66% of EM broken** | composition CLOSED on frozen encoder |
| U2 oracle routing (even with package oracle) | +0.002 | routing cannot rescue |
| Full report | `results/UNIFIED_FINAL_REPORT.md` | |

## 2. G1 — prototype memory (closed)

| metric | value |
|---|---|
| NOVEL | **0.0218** (vs U1 0.0894) |
| Prototype signal | ~2/3 package-local |
| Report | `results/g1/G1_REPORT.md` (commit 7a9d0087) |

## 3. E1 — fine-grained evidence readouts on frozen encoder (closed)

| metric | value |
|---|---|
| Best (E1-B) NOVEL | **0.0693**; PN 0.095 |
| Attention | near-uniform (2.54/2.7 nats) — re-pooling, not evidence-finding |
| Retention | ~20% |
| Verdict | five readout families converge 0.05–0.09: no package-transferable name-primitive structure in frozen encoder |
| Report | `results/e1/E1_REPORT.md` (e1abe7df) |

## 4. E2 — layer-selective evidence (closed; Outcome C)

| metric | value |
|---|---|
| Best (E2-C: CE-trained states, MLP adapter, NO graph layer) | PN **0.087**, clean-7 NOVEL **0.0518**, retention 0.206 |
| Discovery 1 | **Production edge-misalignment bug**: collate offsets vs padded stride scrambled CFG edges in cache/index/E1 paths → production GAT is a de-facto set encoder |
| Discovery 2 | CE-washout hypothesis INVERTED: CE-trained encoder holds MORE concept signal than pre-CE (0.087 vs 0.051) |
| Report | `results/e2/E2_REPORT.md` (5dd35864) |

## 5. P1 v1–v3 — trained generator encoder (all failed)

| run | outcome |
|---|---|
| v1 | all-negative collapse (no pos_weight, single LR) |
| v2 (pos_weight + split LRs) | flatline |
| v3 (LP-FT curriculum) | frozen phase PN 0.041→0.051; unfreeze DEGRADED (0.028 flat) — third end-to-end failure |
| Lesson | trunk fine-tuning collapses universally (5 total failures incl. P1v2-GAT); pre-CE init inferior to CE init |

## 6. BAP enrichment line (parse_bap_v2 → V3e) — built & validated 2026-08-14/15

- **Audit finding:** V3 keeps op-types only; raw .bir holds literals, global-data addresses, def-use chains, arg identity (~27KB semantics → ~600 tags/fn). 936 raw .bir local (4.3GB); enrichment is a local parser pass, no re-lifting.
- **parse_bap_v2** (`src/preprocessing/parse_bap_v2.py`): LIT_* semantic literals (chars, small ints, pow2, magic table), GREF global refs, block-level def-use edges, arg capture, CFG-aware truncation, V3-parity block filter. Validated 299/300 sample parity (O0 outliers = V3's own edge-indexing bug). 395K enriched graphs → `data/graphs_v2` (943 binaries; angie_O3 corrupt everywhere → 76/77 clean-7).
- **Caches:** `results/p1v2` (241,109 train fns; vocab 3,312), `results/p1lex` (+2,000 lex types, vocab 5,312; lex pseudo-block on 43% train / 49% clean-7 fns) — both with built-in verification (row parity, prefix identity, reconstruction).
- **Lexical extraction** (`data/lex_v1`, 834 binaries): .rodata string words + ext-call names per BAP function; scans ALL body constants vs rodata range (graphs_v2 `grefs` alone miss LEA-loaded refs). 135,331 fns with evidence.

## 7. P1v2 head family (closed 2026-08-16) — full report `results/p1v2/P1V2_REPORT.md`

Shared skeleton: frozen CE trunk (base emb rows byte-identical, enrichment/lex rows trainable via row-masked grads, emb wd=0), **no GAT**, no adversarial, MLP adapter + token-conditioned attention over V=5,400 name sub-tokens, BCE pos_weight 5, G1 hard negatives, batch 64, a100_20g.

| variant | input | PN peak | clean-7 ALL | NOVEL | PARTIAL-OOV | AUPRC | retention | gate |
|---|---|---|---|---|---|---|---|---|
| control | V3e, enrichment rows frozen noise (≈V3-only) | 0.0686 | 0.1184 | 0.0297 | 0.0153 | 0.0597 | 0.232 | FAIL |
| embfix | + trained literal/GREF embeddings | 0.0727 | (superseded) | — | — | — | — | — |
| P1-lex | + lexical pseudo-block | 0.0745 | 0.1144 | 0.0324 | 0.0188 | 0.0736 | 0.212 | FAIL |
| **P1-copy** | + direct copy wire `score[t] += copyw·1[t∈lex(fn)]` | **0.0795** | **0.1247** | **0.0385** | **0.0346** | **0.0830** | **0.317** | FAIL |

**Diagnoses that shaped the ladder:**
- GAT run declined (0.028→0.022): random-init GATConv homogenized blocks (pairwise cos 0.02–0.33 → 0.25–0.60) + package-token collapse (`quotearg` in 189/200 top-5s, 52 unique tokens). NOGAT fix → first climbing curve in the P1 family.
- First "enriched" run's 312 enrichment rows were frozen N(0,1) noise (P1_FREEZE froze whole table) — verified std 1.0003/norm 16; fixed with row-masked trainable embeddings.
- Effect-preflight caught AdamW decoupled weight decay shrinking frozen base rows despite zeroed grads → emb param group wd=0.
- GT diagnosis (500 PN fns): P1-lex ≈ embfix paired 45:45 — trunk-fed lexical evidence destroyed before the head (string "bfd" present → `bfd` not predicted). Has-lex fns are 2.3× easier for BOTH models (intrinsic difficulty, not channel effect).

## 8. Direct-evidence rule (banked, zero training) — `experiments_semantic/u1_lex_union.py` (5f343abc)

Rule: when U0∩U1 = ∅ (test-time signal retrieval is lost), union per-binary df<0.2-filtered lexical atoms into U1's prediction. Gate fixed a priori; single-shot clean-7:

| stratum | U1 | U1+lex (gated) | delta | 95% CI |
|---|---|---|---|---|
| NOVEL | 0.0931 | **0.1008** | **+0.0077** | [+0.0047, +0.0107] |
| ALL | 0.3199 | **0.3235** | **+0.0036** | [+0.0028, +0.0043] |
| SEEN | 0.3950 | 0.3960 | +0.0010 | no cost |

Strictly dominant. First real cross-project NOVEL gain over U1 in the arc.

## 9. Measured ceilings (clean-7, atomized)

| quantity | value |
|---|---|
| Novel fns with any lexical evidence | 47% |
| Among those: recall of true name tokens / precision / ≥1 hit | 9.8% / 6.7% / 20% |
| Perfect-exploitation ceiling | ≈ +0.02 NOVEL |
| Achieved (either routing) | ≈ +0.008–0.009 NOVEL |

## 10. Phase conclusions

1. **Evidence path length is the controlling variable** — PN monotone in routing directness (0.0686 → 0.0727 → 0.0745 → 0.0795); trunk-fed evidence is destroyed en route.
2. **The lexical channel ≈ +0.01 NOVEL cross-project regardless of routing**; binding constraint = evidence coverage, not modeling.
3. **Learned composition heads are closed** — dominated by retrieval-composition + direct rules; trunk fine-tuning collapses.
4. Copy-wire signal is more portable (retention 0.317 vs 0.21–0.23) but insufficient.

## 11. Next directions (direct-evidence spirit)

1. Refine U1-union rule: cross-binary df junk filter; string>ext source weighting; dev-tuned partial-disagreement trigger (~+0.010 headroom in transparent sweep).
2. Attack coverage: GREF chasing through .data pointers; caller/callee lexical inheritance (wrappers inherit callee strings).
3. Any future learned component: path-length-1 wire pattern only.

## Key artifacts
- Reports: `results/UNIFIED_FINAL_REPORT.md`, `results/g1/G1_REPORT.md`, `results/e1/E1_REPORT.md`, `results/e2/E2_REPORT.md`, `results/p1v2/P1V2_REPORT.md`, `results/RESEARCH_PROPOSAL_DUAL_HEAD.md`
- Code: `src/preprocessing/parse_bap_v2.py`, `experiments_semantic/{p1_cache_v2,p1_cache_v3,p1_lex_extract,p1v2_train,u1_lex_union}.py`
- HPC: `strlex_ws/results/p1v2/p1_ckpt_{P1-full,P1-embfix,P1-lex,P1-copy}.pt` + fold0 + summaries
- Key commits: 2b51f999, 6f4db119, 1c6de942, 6af5ab9f, 5f343abc, c8918cd7
