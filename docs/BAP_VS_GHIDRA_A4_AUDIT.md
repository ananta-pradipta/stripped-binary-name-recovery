# BAP vs Ghidra capability audit — for the A4 (decompiled-text generation head) decision
2026-08-25. Sources: raw .bir inspection, parse_bir_v3 code, dataset_v2 loader, model wiring,
the 2026-08-14 underutilization audit, RCDG Stage-0 Ghidra feasibility results, A2 census.

## 1. Channel-by-channel: what BAP produces, and what we actually use

| Channel | In raw .bir | Parsed (parse_bir_v3) | Reaches the model | Status |
|---|---|---|---|---|
| Instruction semantics (BIL) | yes | as ~2K type tokens (V3) | yes | core input |
| Real CFG edges | yes | yes (B11 fixed) | yes | core input |
| Call sites: import names + kinds | yes | yes | ext-call encoder + callee/caller sigs | used |
| RETURN vs indirect call | yes | yes (B12 fixed) | yes | used |
| String refs via gref → .rodata | yes | gref_addrs → resolve_string_refs_v2 | **yes since A1a** | the +0.014/+0.031 win |
| Immediates (bucketed `lit_tokens`) | yes | yes, per block | **NO — parsed but never loaded** | UNDERUTILIZED |
| Arg registers / in-args / has_result | yes | yes (`arg_regs_used`, `bap_in_args`, `bap_has_result`) | **NO** | UNDERUTILIZED |
| Written registers | yes | parsed | NO | underutilized (low value) |
| Non-string .rodata (const tables) | reachable via gref_addrs | only strings resolved | NO | **UNTESTED — A2 census killed *immediates* only; magic constants live in exactly these tables** |
| Expression trees / def-use / idioms | yes (~27KB/fn semantics) | collapsed to type tags | NO | underutilized; partial use in P1-lex era (best novel system U1+lex 0.1008 used lexical/idiom features) |
| Type inference (structs, ptrs, sigs) | ABI-level only | – | – | BAP can't (no decompiler-grade types) |
| C-like structured text | **BAP cannot produce** | – | – | the A4 gap |

Verdict on "did we underutilize BAP": **yes, in four concrete places** — (1) `lit_tokens` and (2) arg/ABI
features are parsed and sitting in every graph JSON but never consumed; (3) gref-reachable
.rodata *tables* were never matched against constant lexicons (my A2 census tested instruction
immediates only, so "magic constants dead" was proven only for immediates); (4) expression-level
structure was never given to any model — the Aug-14 conclusion "we tested V3-compression, never
BAP-IR itself" still stands for the encoder.

## 2. What Ghidra adds that BAP structurally cannot
1. **Decompiled C text** — reconstructed loops/conditionals/expressions. BAP has no decompiler;
   BIL is semantically complete but syntactically alien: no pretrained LM has ever seen it, so the
   code-LM naming prior (the mechanism behind SymGen's FT 0.120 vs our 0.037) is unreachable from
   BAP output without pretraining our own LM from scratch on BIL (months, no corpus).
2. **Decompiler-grade type/variable recovery** — signatures, struct fields, named locals.
3. **In-context lexical evidence** — strings/constants/callees at their use sites inside readable code.

## 3. What BAP does better (why the retrieval path stays BAP)
Deterministic, fast (~100× vs decompilation), byte-reproducible corpus (validated), precise flag
semantics for our token scheme, and the whole calibrated retrieval + router stack is built on it.
Ghidra: ~0.01% decomp failures, version-sensitive output, needed a 3-retry workaround for an
11.2.1 race (RCDG worker). RCDG Stage-0 already validated feasibility at corpus scale
(4,728/4,728 fns, 98.9% strict parse) and the P4 pipeline decompiled 714 binaries with 99.99%
function success — so A4's input side is proven infrastructure, not new risk.

## 4. Recommendation (ordered, gated)
1. **A3+ (BAP, cheap, before A4):** one retrain adding the already-parsed unused channels to the
   A1a config — lit_tokens into block token streams, arg-count/has-result/arg-regs as function
   features, plus a rodata-table constant matcher (redeems A2 properly). Gate: > A1a val 0.1310.
2. **A4 (Ghidra text, generation head only):** ~220M code-LM fine-tuned on decompiled functions
   of the v2 train set; BAP retrieval + router unchanged; router arbitrates. Justified because the
   one thing BAP cannot supply is exactly the mechanism with the largest measured FT payoff.
3. Expression-idiom features (BAP) stay a P5-Ablations candidate, not a blocker.
