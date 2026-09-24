# Phase-0 manual findings (2026-08-17)

## B1 collate misalignment — CONFIRMED on dev AND unified
- dev:src/training/train.py collate_fn offsets edge_index by running sum of sample['num_blocks'];
  function_namer.forward flattens block_embs to (B*max_blocks) rows with batch_vec = arange(B).repeat_interleave(max_blocks).
- Regression test scratchpad/tests/test_collate_edges.py FAILS on current code:
  "sample 1: edges [[3,4,7],[4,7,5]] not within node rows [30,60)".
- Fix: offset += max_blocks (block_tokens.size(0)). Same fix needed in any other collate (train_lm_decoder.py, eval scripts, index builders).

## B2 missing CALL_<sym> channel — ROOT CAUSE = stale graphs from an older parse_bap
- data/bir/dash_dash_O2.bir DOES contain named calls inside real subs (sub_158d0: call @sysconf, @times, @intrinsic:...).
- data/graphs/dash_dash_O2_sub_158d0.json: CALL_INTERNAL x21 (wrong); data/graphs_fixed/ and graphs_v2/: CALL_sysconf, CALL_times, CALL_intrinsic x16 (right).
- So the original graphs for dash/gettext/psmisc/grep/sed were produced by a parser version that mapped named calls to CALL_INTERNAL. Fix = full deterministic re-parse with parser-version stamp in every graph JSON + presence invariant.
- SIDE FINDING (V3 loss): BAP FP intrinsics (`call @intrinsic:cast_sfloat_rne_ieee754_binary_64`, `fdiv_rne...`) become CALL_intrinsic (16 in one fn) — should be FP_OP-class tokens; the intrinsic line is a continuation line without address prefix (parser must handle).
- SIDE FINDING: `call #<n> with noreturn` (BAP virtual-var call, 270 in dash O2 vs 92 recutils) currently → CALL_INDIRECT; needs a census of raw statement forms vs V3 mapping.

## PLT/IBT
- All checked ELFs have .plt/.plt.sec (IBT); recutils EXEC (non-PIE), dash/sed DYN (PIE). PIE vs EXEC affects address matching (B3).
