# Phase-0 dataset audit (B2/B3/B5/B6/B8)

Date: 2026-08-17. Repo: $HOME/cs785-project (branch `unified`, read-only). Outputs: this dir only.

## Method / provenance

- **One streaming pass** over all 488,326 files in `data/graphs/` (`scan_graphs.py`, `multiprocessing.Pool(6)`, `json.load` once per file, 32 s wall-clock thanks to page cache). **No subsampling was needed.** 1 file failed to parse (`.gitkeep`).
- Per file recorded: `binary`, `function_name`, is_sub (`function_name.startswith('sub_')`), any token starting with `CALL_` other than `CALL_INTERNAL`/`CALL_INDIRECT` (over `blocks[*].tokens`), token count, block count, md5 of the concatenated token stream (`'|'.join` over all blocks in file order), thunk flag (<=2 tokens and contains `CALL_INTERNAL`) + `internal_callees[0]`.
- 52,698 graph files (older parse format, e.g. `wget_wget_O0_*`, `groff_troff_O*`, `bison_bison_O1/O3`, `tar_tar_O1/O3`) have **no `binary`/`function_name` fields**; binary was inferred from the filename by longest-prefix match against the set of binary ids in match_index ∪ labels ∪ external_calls (0 unresolvable). Files split: 387,482 `sub_*` graphs, 100,843 non-`sub_` graphs.
- `data/match_index.json`: 301,870 entries, 881 binaries; key = graph path, fields `binary,bap_name,real_name,address`.
- `data/labels/<binary>_labels.json`: 945 files. 929 have `name_to_addr`/`addr_to_name` (used `name_to_addr`); 16 (dropbear, pigz, rsync, socat, tengine, tmux) only have `functions` (name→addr). NB the `functions` field is **inconsistent in direction** across files (addr→name in 369 files, name→addr elsewhere) — do not use it blindly. The 70 extra `data/labels/<binary>.json` (no `_labels`) files are addr→name duplicates of the `_labels` files (verified identical for acct_ac_O1/O3) and were ignored.
- `data/external_calls/<binary>_external.json` (942 files): `functions[*].external_calls` (list of `{name,call_order}`); 'with calls' = non-empty list.
- `data/split_assignments.json`: train 431 / val 17 / test 18 / excluded 4 binary ids.
- Loader: `src/preprocessing/build_dataset.py` (unified) vs `git show dev:src/preprocessing/build_dataset.py`: identical iteration/thunk/split logic (unified only adds string-lexicon + retrieval-memory channels). Iterates `self.match_index.items()` (unified L228 / dev L215), skips keys not on disk (L229/L216). Thunk resolution L239-262 (dev L226-249): if <=2 tokens and `CALL_INTERNAL` in tokens, replace blocks/edges with `<binary>_<internal_callees[0]>.json`. `get_splits()` (unified L651-707 / dev L539-): binaries not in train/val/test/excluded → **added to train** (unified L671-674, dev L559-562).
- Binary id parsing: pkg = prefix before first `_`; opt = trailing `_O[0-3]` else `default`.

## B2 — Missing `CALL_<sym>` block-token channel

Definition: over `*_sub_*` graphs only, a function 'has the channel' if any block token starts with `CALL_` and is not `CALL_INTERNAL`/`CALL_INDIRECT`.

- Overall: **0.415** of all 387,482 sub_ graphs; **0.436** of the matched (match_index) sub_ graphs.
- By opt level (all sub_ graphs): O0: 0.242 (n=211,433), O1: 0.575 (n=51,138), O2: 0.603 (n=59,578), O3: 0.680 (n=47,242), default: 0.684 (n=18,091). (O0 is low because ~half of O0 sub_ graphs are 1-token ENDBR64 thunks whose CALL_INTERNAL is the only token.)
- Binaries with sub_ graphs and **0% channel: 53 / 932**. Flagged (external.json ≥10 functions with calls AND channel 0%): **53 binaries**, all listed below.
- Mechanism (spot-checked): In flagged binaries PLT calls are emitted as CALL_INTERNAL/CALL_INDIRECT and no block has has_external_call=True (checked dash_dash_O2 sub_102f0/sub_10390, grep_grep_O2, psmisc_pstree_O2 samples): the BAP-IR for these binaries did not resolve PLT symbol names, so the graph channel is empty even though <binary>_external.json lists calls. (Cross-ref: NOTES_manual.md in this dir independently traced this to a stale parser version — `data/bir/dash_dash_O2.bir` has `call @sysconf` etc. and `data/graphs_fixed/`/`graphs_v2/` re-parses carry `CALL_sysconf`.)

### Flagged binaries (ext.json ≥10 fns with calls, graph channel 0%)

| binary | #sub_ graphs | #matched | ext.json fns with calls (sub_ only) |
|---|---|---|---|
| cvs_cvs_O0 | 1378 | 0 | 1078 (842) |
| cvs_cvs_O1 | 844 | 0 | 801 (573) |
| cvs_cvs_O2 | 778 | 0 | 789 (556) |
| cvs_cvs_O3 | 722 | 0 | 776 (540) |
| dash_dash_O0 | 676 | 348 | 236 (153) |
| dash_dash_O1 | 255 | 254 | 212 (129) |
| dash_dash_O2 | 231 | 229 | 214 (132) |
| dash_dash_O3 | 221 | 219 | 225 (143) |
| gettext_msgfmt_O0 | 155 | 80 | 198 (64) |
| gettext_msgfmt_O1 | 58 | 56 | 177 (46) |
| gettext_msgfmt_O2 | 65 | 62 | 182 (50) |
| gettext_msgfmt_O3 | 58 | 55 | 178 (46) |
| gettext_msgmerge_O0 | 45 | 30 | 128 (19) |
| gettext_msgmerge_O1 | 11 | 11 | 119 (10) |
| gettext_msgmerge_O2 | 14 | 13 | 122 (12) |
| gettext_msgmerge_O3 | 21 | 13 | 125 (15) |
| gettext_xgettext_O0 | 968 | 556 | 419 (257) |
| gettext_xgettext_O1 | 307 | 307 | 334 (181) |
| gettext_xgettext_O2 | 351 | 338 | 372 (218) |
| gettext_xgettext_O3 | 334 | 313 | 371 (217) |
| grep_grep_O0 | 990 | 493 | 326 (218) |
| grep_grep_O1 | 275 | 274 | 267 (156) |
| grep_grep_O2 | 253 | 250 | 285 (173) |
| grep_grep_O3 | 252 | 250 | 302 (190) |
| libyaml_run-parser_O1 | 1 | 1 | 13 (0) |
| lighttpd_lighttpd_O0 | 817 | 0 | 463 (156) |
| lighttpd_lighttpd_O1 | 252 | 0 | 392 (82) |
| lighttpd_lighttpd_O2 | 242 | 0 | 402 (84) |
| lighttpd_lighttpd_O3 | 229 | 0 | 423 (87) |
| psmisc_killall_O0 | 46 | 23 | 84 (22) |
| psmisc_killall_O1 | 9 | 9 | 70 (7) |
| psmisc_killall_O2 | 9 | 8 | 70 (7) |
| psmisc_killall_O3 | 9 | 8 | 70 (7) |
| psmisc_peekfd_O0 | 12 | 6 | 28 (5) |
| psmisc_peekfd_O1 | 6 | 6 | 28 (3) |
| psmisc_peekfd_O2 | 7 | 6 | 28 (3) |
| psmisc_peekfd_O3 | 6 | 5 | 27 (2) |
| psmisc_prtstat_O0 | 22 | 11 | 32 (9) |
| psmisc_prtstat_O1 | 5 | 5 | 27 (3) |
| psmisc_prtstat_O2 | 6 | 5 | 27 (3) |
| psmisc_prtstat_O3 | 4 | 3 | 26 (2) |
| psmisc_pstree_O0 | 79 | 40 | 87 (32) |
| psmisc_pstree_O1 | 20 | 20 | 68 (14) |
| psmisc_pstree_O2 | 15 | 14 | 68 (13) |
| psmisc_pstree_O3 | 26 | 25 | 79 (24) |
| sed_sed_O0 | 667 | 331 | 258 (165) |
| sed_sed_O1 | 193 | 192 | 202 (108) |
| sed_sed_O2 | 159 | 154 | 199 (105) |
| sed_sed_O3 | 183 | 171 | 231 (138) |
| tinycc_tcc_O0 | 526 | 0 | 195 (128) |
| tinycc_tcc_O1 | 378 | 0 | 172 (107) |
| tinycc_tcc_O2 | 329 | 0 | 178 (113) |
| tinycc_tcc_O3 | 334 | 0 | 196 (131) |

By package: psmisc 16, gettext 12, cvs 4, dash 4, grep 4, lighttpd 4, sed 4, tinycc 4, libyaml 1 (`libyaml_run-parser_O1`, only 3 sub_ graphs). **dash/gettext/psmisc/grep/sed are in match_index (i.e. trained/evaluated on) with an empty ext-call channel; cvs/lighttpd/tinycc are unmatched (see B3).**

### Per-package table (sorted by channel presence; sub_ graphs only)

| package | frac sub_ w/ CALL_sym | #sub_ graphs | frac matched w/ CALL_sym | #matched | #binaries | #flagged | ext.json fns w/ calls |
|---|---|---|---|---|---|---|---|
| angie | n/a | 0 | n/a | 0 | 4 | 0 | 2,749 |
| nginx118 | n/a | 0 | n/a | 0 | 4 | 0 | 2,504 |
| zstd | n/a | 0 | n/a | 0 | 1 | 0 | 0 |
| cvs | 0.000 | 3,722 | n/a | 0 | 4 | 4 | 3,444 |
| dash | 0.000 | 1,383 | 0.000 | 1,050 | 4 | 4 | 887 |
| gettext | 0.000 | 2,387 | 0.000 | 1,834 | 12 | 12 | 2,725 |
| grep | 0.000 | 1,770 | 0.000 | 1,267 | 4 | 4 | 1,180 |
| lighttpd | 0.000 | 1,540 | n/a | 0 | 4 | 4 | 1,680 |
| psmisc | 0.000 | 281 | 0.000 | 194 | 16 | 16 | 819 |
| sed | 0.000 | 1,202 | 0.000 | 848 | 4 | 4 | 890 |
| tinycc | 0.000 | 1,567 | n/a | 0 | 4 | 4 | 741 |
| curl | 0.138 | 14,784 | 0.195 | 10,452 | 4 | 0 | 3,079 |
| libxml2 | 0.219 | 7,497 | 0.187 | 3,846 | 4 | 0 | 2,108 |
| libsodium | 0.225 | 6,028 | 0.173 | 2,263 | 10 | 0 | 1,412 |
| xz | 0.227 | 2,221 | 0.123 | 609 | 6 | 0 | 744 |
| expat | 0.236 | 1,108 | 0.478 | 157 | 4 | 0 | 476 |
| libpng | 0.254 | 1,383 | 0.260 | 1,046 | 4 | 0 | 571 |
| zlib | 0.256 | 4,312 | 0.336 | 2,162 | 8 | 0 | 302 |
| strace | 0.269 | 9,346 | 0.351 | 6,961 | 4 | 0 | 2,943 |
| tmux | 0.303 | 6,121 | 0.236 | 1,949 | 2 | 0 | 2,122 |
| sqlite | 0.307 | 15,118 | 0.307 | 15,078 | 4 | 0 | 4,689 |
| wget | 0.317 | 1,536 | n/a | 0 | 1 | 0 | 619 |
| lua | 0.326 | 5,909 | 0.326 | 5,897 | 8 | 0 | 1,950 |
| binutils2 | 0.354 | 39,154 | 0.336 | 28,888 | 16 | 0 | 14,818 |
| less | 0.369 | 3,228 | 0.370 | 3,194 | 15 | 0 | 1,714 |
| jq | 0.370 | 4,911 | 0.374 | 4,775 | 4 | 0 | 2,097 |
| mailutils | 0.371 | 1,782 | 0.023 | 946 | 8 | 0 | 1,426 |
| pcre2 | 0.372 | 1,103 | 0.377 | 485 | 8 | 0 | 964 |
| lz4 | 0.385 | 1,382 | 0.394 | 1,227 | 4 | 0 | 728 |
| dropbear | 0.388 | 4,377 | 0.360 | 2,255 | 6 | 0 | 2,221 |
| binutils | 0.390 | 63,079 | 0.396 | 58,493 | 30 | 0 | 26,449 |
| ed | 0.392 | 605 | 0.360 | 464 | 4 | 0 | 434 |
| rsync | 0.404 | 2,403 | 0.314 | 974 | 2 | 0 | 1,270 |
| libarchive | 0.412 | 14,691 | 0.499 | 6,594 | 12 | 0 | 7,403 |
| cflow | 0.423 | 1,550 | n/a | 0 | 4 | 0 | 913 |
| libyaml | 0.424 | 738 | 0.564 | 298 | 8 | 1 | 492 |
| pigz | 0.432 | 493 | 0.342 | 114 | 2 | 0 | 383 |
| indent | 0.432 | 400 | 0.435 | 395 | 4 | 0 | 338 |
| csplit2 | 0.441 | 1,555 | 0.440 | 1,544 | 4 | 0 | 961 |
| gzip | 0.460 | 819 | 0.461 | 805 | 5 | 0 | 720 |
| inetutils2 | 0.471 | 1,261 | 0.424 | 865 | 12 | 0 | 1,281 |
| screen | 0.475 | 3,469 | 0.475 | 3,441 | 5 | 0 | 2,219 |
| gnuchess | 0.479 | 2,164 | 0.476 | 2,128 | 4 | 0 | 1,312 |
| cppi | 0.481 | 428 | 0.482 | 421 | 4 | 0 | 434 |
| bison | 0.482 | 6,739 | 0.491 | 6,499 | 5 | 0 | 3,586 |
| direvent | 0.488 | 2,193 | 0.488 | 2,175 | 4 | 0 | 1,524 |
| bzip2 | 0.500 | 396 | 0.497 | 388 | 4 | 0 | 341 |
| patch | 0.504 | 1,986 | 0.505 | 1,978 | 5 | 0 | 1,513 |
| busybox | 0.504 | 15,984 | 0.465 | 11,739 | 4 | 0 | 8,062 |
| flex | 0.505 | 1,304 | 0.507 | 1,292 | 5 | 0 | 986 |
| rcs | 0.505 | 1,767 | 0.506 | 1,751 | 4 | 0 | 1,184 |
| gawk | 0.510 | 4,949 | 0.512 | 4,770 | 5 | 0 | 3,188 |
| nettle | 0.516 | 213 | 0.471 | 157 | 8 | 0 | 287 |
| rush | 0.524 | 2,614 | 0.524 | 2,526 | 4 | 0 | 1,832 |
| cpio | 0.528 | 1,938 | 0.529 | 1,930 | 5 | 0 | 1,588 |
| coreutils4 | 0.530 | 15,002 | 0.535 | 10,813 | 124 | 0 | 14,320 |
| acct | 0.535 | 1,210 | 0.537 | 1,164 | 24 | 0 | 1,456 |
| inetutils | 0.537 | 3,383 | 0.539 | 3,346 | 19 | 0 | 3,558 |
| coreutils3 | 0.538 | 9,051 | 0.538 | 6,568 | 68 | 0 | 8,628 |
| tar | 0.539 | 5,476 | 0.540 | 5,445 | 5 | 0 | 3,775 |
| socat | 0.539 | 1,532 | 0.407 | 747 | 2 | 0 | 1,274 |
| coreutils2 | 0.541 | 9,082 | 0.543 | 8,979 | 45 | 0 | 8,151 |
| bc | 0.544 | 890 | 0.532 | 855 | 6 | 0 | 643 |
| diffutils2 | 0.550 | 3,225 | 0.535 | 2,342 | 16 | 0 | 3,056 |
| findutils | 0.559 | 3,741 | 0.560 | 3,713 | 10 | 0 | 3,071 |
| m4 | 0.569 | 3,580 | 0.571 | 3,384 | 5 | 0 | 2,557 |
| nginx | 0.571 | 2,930 | 0.572 | 2,925 | 2 | 0 | 1,389 |
| coreutils | 0.572 | 17,235 | 0.574 | 17,016 | 108 | 0 | 16,442 |
| tengine | 0.575 | 3,403 | n/a | 0 | 2 | 0 | 2,032 |
| plotutils | 0.587 | 1,115 | 0.554 | 835 | 28 | 0 | 2,110 |
| time | 0.613 | 62 | 0.648 | 54 | 5 | 0 | 182 |
| wdiff | 0.648 | 165 | 0.593 | 118 | 4 | 0 | 322 |
| htop | 0.650 | 1,771 | 0.660 | 1,659 | 4 | 0 | 1,751 |
| tree | 0.651 | 481 | 0.652 | 460 | 4 | 0 | 517 |
| enscript | 0.669 | 508 | 0.674 | 500 | 5 | 0 | 686 |
| gperf | 0.673 | 565 | 0.670 | 545 | 5 | 0 | 460 |
| recutils | 0.681 | 2,816 | n/a | 0 | 36 | 0 | 6,651 |
| bash | 0.684 | 6,141 | 0.685 | 6,086 | 5 | 0 | 6,614 |
| spell | 0.690 | 58 | 0.731 | 52 | 4 | 0 | 175 |
| units | 0.700 | 799 | 0.703 | 791 | 5 | 0 | 875 |
| groff | 0.712 | 9,301 | 0.701 | 8,803 | 28 | 0 | 5,655 |
| which | 0.720 | 150 | 0.739 | 142 | 5 | 0 | 297 |
| dos2unix | 0.725 | 466 | 0.730 | 460 | 8 | 0 | 644 |
| make | 0.740 | 1,090 | 0.744 | 1,085 | 5 | 0 | 1,934 |
| nano | 0.748 | 400 | 0.748 | 397 | 1 | 0 | 453 |
| texinfo | 0.775 | 1,600 | 0.781 | 1,568 | 8 | 0 | 1,932 |
| datamash | 0.822 | 808 | 0.825 | 800 | 4 | 0 | 928 |
| combinatorics | 0.886 | 220 | 0.898 | 215 | 3 | 0 | 325 |
| hello | 0.964 | 336 | 0.975 | 324 | 4 | 0 | 536 |

(angie/nginx118/zstd have external.json/labels but no graphs at all in data/graphs. Full per-binary numbers in audit.json → B2.per_binary.)

## B3 — Address matching / silent drops

Per binary: #label functions (unique addresses from `name_to_addr`; also #names) vs #match_index entries vs #`_sub_` graph files. match rate = matched / unique label addrs.

- 945 binaries with labels, 881 in match_index, 932 with sub_ graphs. Totals: 400,892 unique label addrs, 301,870 matched entries, 387,482 sub_ graphs.
- Match rate: median **0.723**, mean 0.809. Per opt (median): O0 0.998, O1 0.712, O2 0.643, O3 0.698, default 0.686. **715 / 942 binaries have match rate < 0.90** (50 O0, 195 O1, 209 O2, 201 O3, 60 default) — i.e. essentially every non-O0 binary.
- O0 rates >1 (up to 1.99): the matcher matched **both** the 1-token ENDBR64 thunk `sub_X` and the body `sub_X+4` to the same label. Extra duplicate (same binary, same real_name) match_index entries: O0 55,979, O2 1,286, O1 601, O3 475, default 26 = **58,367 duplicate samples** (19% of match_index; after loader thunk resolution these are token-identical duplicates).
- Why labels are unmatched (label addr vs sub_ graph at addr or addr+4, or name already matched):
| opt | matched | label has NO sub_ graph at addr/addr+4 | sub_ graph exists but unmatched |
|---|---|---|---|
| O0 | 86,272 | 3,153 | 124 |
| O1 | 48,148 | 16,184 | 8 |
| O2 | 48,724 | 21,926 | 7 |
| O3 | 44,102 | 18,407 | 8 |
| default | 17,520 | 8,762 | 0 |
  So the drop is almost entirely 'BAP produced no `sub_` graph at the label address' — not an off-by-N matcher bug. Top unmatched names at O1-O3: `main` (~200 binaries each), `__do_global_dtors_aux/frame_dummy/register_tm_clones/_init/_fini/_start` (crt junk, harmless), and gnulib `xcalloc/xizalloc/xinmalloc/version_etc_ar/quotearg_*/setlocale_null` (~90-105 binaries each).
- **`main` is silently dropped**: of 942 binaries with a `main` label, only 111 have `main` in match_index; in 813 of the 831 unmatched cases a BAP graph named `<binary>_main.json` exists (703 of them with ≥10 tokens, i.e. a real body) — BAP recovered main by symbol/heuristic, but the matcher only considers `sub_<addr>` graphs (its `address` field is a BAP tid, not the label address).
- Same pattern for other symbol-named graphs: 16,953 additional (binary,label) pairs have a `<binary>_<label>.json` graph but no match (14,096 with ≥10 tokens); top names: xmalloc (44), xrealloc (44), argp_error (43), argp_failure (43), argp_help (43), argp_state_help (43), xcalloc (40), xstrdup (40). (Mostly bash -rdynamic exports and gnulib/argp; some may be PLT stubs, so treat as an upper bound.)
- **Zero-match binaries (labels + sub_ graphs, 0 match_index entries): 55**: cflow_cflow_O0, cflow_cflow_O1, cflow_cflow_O2, cflow_cflow_O3, cvs_cvs_O0, cvs_cvs_O1, cvs_cvs_O2, cvs_cvs_O3, lighttpd_lighttpd_O0, lighttpd_lighttpd_O1, lighttpd_lighttpd_O2, lighttpd_lighttpd_O3, recutils_csv2rec_O0, recutils_csv2rec_O1, recutils_csv2rec_O2, recutils_csv2rec_O3, recutils_rec2csv_O0, recutils_rec2csv_O1, recutils_rec2csv_O2, recutils_rec2csv_O3, recutils_recdel_O0, recutils_recdel_O1, recutils_recdel_O2, recutils_recdel_O3, recutils_recfix_O0, recutils_recfix_O1, recutils_recfix_O2, recutils_recfix_O3, recutils_recfmt_O0, recutils_recfmt_O1, recutils_recfmt_O2, recutils_recfmt_O3, recutils_recinf_O0, recutils_recinf_O1, recutils_recinf_O2, recutils_recinf_O3, recutils_recins_O0, recutils_recins_O1, recutils_recins_O2, recutils_recins_O3, recutils_recsel_O0, recutils_recsel_O1, recutils_recsel_O2, recutils_recsel_O3, recutils_recset_O0, recutils_recset_O1, recutils_recset_O2, recutils_recset_O3, tengine_nginx_O0, tengine_nginx_O2, tinycc_tcc_O0, tinycc_tcc_O1, tinycc_tcc_O2, tinycc_tcc_O3, wget_wget_O0.
- match_index binaries with no labels file at all: ['diffutils_cmp', 'diffutils_diff', 'diffutils_diff3', 'diffutils_sdiff'] (labels came from elsewhere).

### Zero-match offset characterisation (label addr → nearest `sub_` graph addr)

| binary | #labels | #sub_ graphs | frac label addr with sub_ graph at exactly addr | at addr+4 | median |Δ| | in split file |
|---|---|---|---|---|---|---|
| cvs_cvs_O2 | 1162 | 778 | 0.664 | 0.001 | 0 | — |
| recutils_csv2rec_O0 | 180 | 237 | 0.361 | 0.600 | 4 | — |
| tinycc_tcc_O0 | 514 | 526 | 0.984 | 0.000 | 0 | — |
| lighttpd_lighttpd_O2 | 869 | 242 | 0.261 | 0.000 | 272 | — |
| cflow_cflow_O2 | 317 | 239 | 0.744 | 0.003 | 0 | — |
| tinycc_tcc_O2 | 369 | 329 | 0.883 | 0.000 | 0 | — |
| lighttpd_lighttpd_O0 | 1070 | 817 | 0.579 | 0.000 | 0 | — |
| cflow_cflow_O1 | 326 | 263 | 0.801 | 0.003 | 0 | — |
| cflow_cflow_O3 | 294 | 214 | 0.714 | 0.003 | 0 | — |
| cflow_cflow_O0 | 424 | 834 | 0.969 | 0.031 | 0 | — |
| cvs_cvs_O3 | 1105 | 722 | 0.644 | 0.001 | 0 | — |
| cvs_cvs_O1 | 1188 | 844 | 0.705 | 0.000 | 0 | — |
| tengine_nginx_O2 | 569 | 590 | 0.000 | 0.000 | 1728 | — |
| tengine_nginx_O0 | 569 | 2813 | 0.000 | 1.000 | 4 | — |
| wget_wget_O0 | 777 | 1536 | 0.991 | 0.001 | 0 | ['train'] |
| lighttpd_lighttpd_O3 | 864 | 229 | 0.247 | 0.000 | 301 | — |
| recutils_recsel_O2 | 133 | 41 | 0.286 | 0.008 | 672 | — |
| lighttpd_lighttpd_O1 | 764 | 252 | 0.323 | 0.000 | 175 | — |
| cvs_cvs_O0 | 1419 | 1378 | 0.960 | 0.000 | 0 | — |

Sampled 10 label addresses each for cflow_cflow_O0/O1/O2/O3 and cvs_cvs_O0 (see audit.json → B3.zero_match_offset_samples): every sampled delta is **0** — the graphs sit exactly at the label addresses. So for cflow/cvs/tinycc/wget_wget_O0 (and partly lighttpd/recutils) the address matcher was simply **never run / match_index never rebuilt** for these binaries; **tengine_nginx_O0** is a pure +4 (ENDBR64) offset for 100% of labels (matcher would need the +4 rule), and **tengine_nginx_O2** has 0% exact/+4 hits with median |Δ|=1728 → different build than the labels (consistent with the known ELF-vs-BIR build mismatch). `wget_wget_O0` is listed in split `train` but contributes **0 samples**.

### Binaries with match rate < 0.90 (715)

Full list with numbers in audit.json → B3.bins_rate_lt_090. Lowest 30 (excluding the 55 zero-match binaries):

| binary | labels(addrs) | matched | sub_ graphs | rate |
|---|---|---|---|---|
| expat_xmlwf_O2 | 40 | 3 | 189 | 0.075 |
| libyaml_run-parser_O1 | 8 | 1 | 1 | 0.125 |
| xz_lzmainfo_O2 | 42 | 7 | 45 | 0.167 |
| bash_bash_O3 | 2249 | 460 | 478 | 0.205 |
| mailutils_messages_O0 | 9 | 2 | 4 | 0.222 |
| mailutils_messages_O2 | 9 | 2 | 4 | 0.222 |
| bash_bash_O2 | 2309 | 520 | 528 | 0.225 |
| bash_bash | 2309 | 521 | 528 | 0.226 |
| bash_bash_O1 | 2364 | 576 | 578 | 0.244 |
| libyaml_run-parser_O3 | 8 | 2 | 2 | 0.250 |
| acct_accton_O3 | 18 | 5 | 7 | 0.278 |
| htop_htop_O3 | 591 | 166 | 175 | 0.281 |
| htop_htop_O2 | 595 | 180 | 183 | 0.303 |
| make_make_O3 | 371 | 117 | 118 | 0.315 |
| htop_htop_O1 | 622 | 200 | 207 | 0.322 |
| make_make | 375 | 123 | 124 | 0.328 |
| make_make_O2 | 375 | 123 | 124 | 0.328 |
| libsodium_keygen_O2 | 314 | 108 | 256 | 0.344 |
| make_make_O1 | 372 | 128 | 129 | 0.344 |
| nettle_nettle-hash_O1 | 14 | 5 | 6 | 0.357 |
| libyaml_run-emitter_O1 | 11 | 4 | 4 | 0.364 |
| libyaml_run-emitter_O3 | 11 | 4 | 4 | 0.364 |
| libpng_pngtest_O1 | 19 | 7 | 7 | 0.368 |
| libsodium_pwhash_scrypt_O2 | 240 | 90 | 269 | 0.375 |
| acct_accton_O1 | 18 | 7 | 9 | 0.389 |
| nettle_nettle-hash_O2 | 15 | 6 | 7 | 0.400 |
| nettle_nettle-hash_O3 | 15 | 6 | 7 | 0.400 |
| libsodium_scalarmult_ed25519_O2 | 134 | 54 | 202 | 0.403 |
| nginx_nginx_O2 | 1224 | 503 | 507 | 0.411 |
| libpng_pngtest_O3 | 17 | 7 | 7 | 0.412 |

## B5 — Duplicate builds mislabelled as optimisation levels

identity = fraction of shared real_names (via match_index) whose full block-token stream (all blocks concatenated, order preserved) has identical md5. identity_thunk_resolved replaces a 1-2-token CALL_INTERNAL thunk graph by its callee graph, mirroring the loader. identity_resolved_ge10tok restricts to functions with >=10 tokens in either build.

- 214 (pkg,bin) groups with ≥2 opt levels → 1416 opt pairs, 1396 with ≥1 shared name. **310 pairs have identity ≥ 0.5.**
- Median identity per opt pair (thunk-resolved): O0-O1 0.010 (n=187), O0-O2 0.013 (n=212), O0-O3 0.014 (n=187), O0-default 0.007 (n=60), O1-O2 0.047 (n=188), O1-O3 0.044 (n=194), O1-default 0.029 (n=60), O2-O3 0.576 (n=188), O2-default 0.790 (n=60), O3-default 0.554 (n=60)
- Distribution (thunk-resolved identity, pairs with shared names): [0,0.05): 813, [0.05,0.1): 133, [0.1,0.2): 45, [0.2,0.3): 19, [0.3,0.4): 34, [0.4,0.5): 42, [0.5,0.6): 89, [0.6,0.7): 79, [0.7,0.8): 60, [0.8,0.9): 38, [0.9,0.95): 20, [0.95,1.0]: 24
- Raw (unresolved) distribution: [0,0.05): 778, [0.05,0.1): 154, [0.1,0.2): 57, [0.2,0.3): 17, [0.3,0.4): 36, [0.4,0.5): 41, [0.5,0.6): 88, [0.6,0.7): 83, [0.7,0.8): 58, [0.8,0.9): 36, [0.9,0.95): 20, [0.95,1.0]: 28
- Interpretation: O2-O3 (median 0.58) and O2-`default` (0.79) being high is expected (default build = O2 for most GNU packages: e.g. bash_bash_O2 vs bash_bash 0.998, binutils_size 0.971, inetutils_telnet 1.000). **Not expected** and indicating a duplicate/mislabelled build: O0 vs O1/O2/O3 ≥0.9 (groff_* O0-O2 0.93-0.98 across troff/pic/refer/tbl/groff, datamash O0-O3 1.000, hello O0-O1-O2 1.000, inetutils_traceroute O0-O2 0.981) and O1-O3 = 1.0 (gperf O1=O2=O3 1.000, inetutils_traceroute, texinfo_ginstall-info, strace 0.922). O0 vs anything is otherwise ~0.01 (e.g. bash, coreutils_ls, strace 0.001), so O0-O2 ≥ 0.9 cannot be compiler coincidence.

### All pairs with identity ≥ 0.5 (thunk-resolved), sorted desc

| bin1 | bin2 | shared names | identity raw | identity resolved | identity (≥10 tok) | n(≥10 tok) |
|---|---|---|---|---|---|---|
| datamash_datamash_O0 | datamash_datamash_O3 | 200 | 1.000 | 1.000 | 1.000 | 178 |
| gperf_gperf_O1 | gperf_gperf_O2 | 105 | 1.000 | 1.000 | 1.000 | 87 |
| gperf_gperf_O1 | gperf_gperf_O3 | 105 | 1.000 | 1.000 | 1.000 | 87 |
| gperf_gperf_O2 | gperf_gperf_O3 | 105 | 1.000 | 1.000 | 1.000 | 87 |
| hello_hello_O0 | hello_hello_O1 | 81 | 1.000 | 1.000 | 1.000 | 67 |
| hello_hello_O0 | hello_hello_O2 | 81 | 1.000 | 1.000 | 1.000 | 67 |
| hello_hello_O1 | hello_hello_O2 | 81 | 1.000 | 1.000 | 1.000 | 67 |
| inetutils_telnet_O2 | inetutils_telnet | 184 | 1.000 | 1.000 | 1.000 | 178 |
| inetutils_traceroute_O1 | inetutils_traceroute_O3 | 54 | 1.000 | 1.000 | 1.000 | 53 |
| libyaml_run-parser_O1 | libyaml_run-parser_O3 | 1 | 1.000 | 1.000 | 1.000 | 1 |
| mailutils_messages_O0 | mailutils_messages_O2 | 2 | 1.000 | 1.000 | 1.000 | 1 |
| pcre2_pcre2test_O2 | pcre2_pcre2test_O3 | 1 | 1.000 | 1.000 | 1.000 | 1 |
| texinfo_ginstall-info_O1 | texinfo_ginstall-info_O3 | 40 | 1.000 | 1.000 | 1.000 | 40 |
| bash_bash_O2 | bash_bash | 520 | 0.998 | 0.998 | 0.998 | 510 |
| inetutils_traceroute_O0 | inetutils_traceroute_O2 | 54 | 0.981 | 0.981 | 0.981 | 53 |
| groff_troff_O0 | groff_troff_O2 | 985 | 0.981 | 0.980 | 0.979 | 917 |
| groff_pic_O0 | groff_pic_O2 | 255 | 0.973 | 0.973 | 0.970 | 234 |
| binutils_size_O2 | binutils_size | 1217 | 0.971 | 0.971 | 0.970 | 1161 |
| groff_refer_O1 | groff_refer_O3 | 239 | 0.971 | 0.967 | 0.968 | 217 |
| groff_refer_O0 | groff_refer_O2 | 234 | 0.966 | 0.966 | 0.967 | 212 |
| groff_eqn_O1 | groff_eqn_O3 | 289 | 0.969 | 0.965 | 0.967 | 273 |
| groff_tbl_O0 | groff_tbl_O2 | 194 | 0.964 | 0.964 | 0.958 | 167 |
| groff_pic_O1 | groff_pic_O3 | 260 | 0.965 | 0.962 | 0.962 | 238 |
| coreutils2_du_O2 | coreutils2_du | 169 | 0.947 | 0.953 | 0.945 | 145 |
| groff_tbl_O1 | groff_tbl_O3 | 199 | 0.955 | 0.950 | 0.947 | 171 |
| cpio_cpio_O2 | cpio_cpio | 243 | 0.947 | 0.947 | 0.938 | 210 |
| coreutils_cut_O2 | coreutils_cut | 51 | 0.941 | 0.941 | 0.933 | 45 |
| coreutils2_mktemp_O2 | coreutils2_mktemp | 49 | 0.918 | 0.939 | 0.930 | 43 |
| coreutils_tr_O2 | coreutils_tr | 63 | 0.937 | 0.937 | 0.925 | 53 |
| coreutils_rm_O2 | coreutils_rm | 124 | 0.935 | 0.935 | 0.921 | 101 |
| coreutils_paste_O2 | coreutils_paste | 46 | 0.935 | 0.935 | 0.925 | 40 |
| groff_groff_O0 | groff_groff_O2 | 133 | 0.932 | 0.932 | 0.927 | 124 |
| coreutils2_sort_O2 | coreutils2_sort | 171 | 0.924 | 0.930 | 0.922 | 154 |
| coreutils2_df_O2 | coreutils2_df | 129 | 0.922 | 0.922 | 0.917 | 108 |
| strace_strace_O1 | strace_strace_O3 | 1018 | 0.925 | 0.922 | 0.923 | 993 |
| flex_flex_O2 | flex_flex | 199 | 0.920 | 0.920 | 0.918 | 195 |
| coreutils_join_O2 | coreutils_join | 74 | 0.919 | 0.919 | 0.908 | 65 |
| coreutils_od_O2 | coreutils_od | 97 | 0.918 | 0.918 | 0.901 | 81 |
| groff_groff_O1 | groff_groff_O3 | 138 | 0.913 | 0.906 | 0.906 | 128 |
| datamash_datamash_O0 | datamash_datamash_O1 | 200 | 0.905 | 0.905 | 0.893 | 178 |
| datamash_datamash_O1 | datamash_datamash_O3 | 200 | 0.905 | 0.905 | 0.893 | 178 |
| gnuchess_gnuchess_O0 | gnuchess_gnuchess_O1 | 532 | 0.906 | 0.904 | 0.897 | 484 |
| findutils_xargs_O2 | findutils_xargs | 102 | 0.902 | 0.902 | 0.886 | 88 |
| groff_eqn_O0 | groff_eqn_O2 | 284 | 0.901 | 0.901 | 0.896 | 269 |
| coreutils3_dirname_O2 | coreutils3_dirname_O3 | 47 | 0.894 | 0.894 | 0.878 | 41 |
| mailutils_mail_O0 | mailutils_mail_O2 | 378 | 1.000 | 0.892 | 1.000 | 9 |
| datamash_datamash_O0 | datamash_datamash_O2 | 200 | 0.885 | 0.885 | 0.871 | 178 |
| datamash_datamash_O1 | datamash_datamash_O2 | 200 | 0.885 | 0.885 | 0.871 | 178 |
| datamash_datamash_O2 | datamash_datamash_O3 | 200 | 0.885 | 0.885 | 0.871 | 178 |
| coreutils4_whoami_O2 | coreutils4_whoami_O3 | 46 | 0.870 | 0.870 | 0.850 | 40 |
| coreutils4_tty_O2 | coreutils4_tty_O3 | 44 | 0.864 | 0.864 | 0.842 | 38 |
| coreutils_fmt_O2 | coreutils_fmt | 58 | 0.862 | 0.862 | 0.840 | 50 |
| coreutils_uniq_O2 | coreutils_uniq | 65 | 0.862 | 0.862 | 0.830 | 53 |
| binutils_readelf_O2 | binutils_readelf | 778 | 0.862 | 0.861 | 0.857 | 747 |
| coreutils3_expand_O2 | coreutils3_expand_O3 | 57 | 0.860 | 0.860 | 0.843 | 51 |
| make_make_O2 | make_make | 120 | 0.858 | 0.858 | 0.848 | 112 |
| coreutils_tsort_O2 | coreutils_tsort | 56 | 0.857 | 0.857 | 0.837 | 49 |
| groff_troff_O1 | groff_troff_O2 | 990 | 0.857 | 0.855 | 0.846 | 921 |
| libarchive_bsdcat_O2 | libarchive_bsdcat_O3 | 48 | 0.875 | 0.854 | 0.854 | 41 |
| groff_soelim_O0 | groff_soelim_O2 | 47 | 0.851 | 0.851 | 0.833 | 42 |
| mailutils_frm_O0 | mailutils_frm_O2 | 20 | 1.000 | 0.850 | 1.000 | 1 |
| groff_troff_O0 | groff_troff_O1 | 985 | 0.849 | 0.847 | 0.838 | 917 |
| coreutils_nl_O2 | coreutils_nl | 52 | 0.846 | 0.846 | 0.818 | 44 |
| coreutils2_ls_O2 | coreutils2_ls | 179 | 0.832 | 0.838 | 0.816 | 158 |
| coreutils2_stty_O2 | coreutils2_stty | 66 | 0.833 | 0.833 | 0.821 | 56 |
| less_less_O2 | less_less | 432 | 0.833 | 0.833 | 0.817 | 394 |
| mailutils_readmsg_O0 | mailutils_readmsg_O2 | 30 | 1.000 | 0.833 | 1.000 | 5 |
| nettle_nettle-hash_O2 | nettle_nettle-hash_O3 | 6 | 0.833 | 0.833 | 0.833 | 6 |
| gperf_gperf_O1 | gperf_gperf | 105 | 0.857 | 0.829 | 0.828 | 87 |
| gperf_gperf_O2 | gperf_gperf | 105 | 0.857 | 0.829 | 0.828 | 87 |
| gperf_gperf_O3 | gperf_gperf | 105 | 0.857 | 0.829 | 0.828 | 87 |
| texinfo_ginstall-info_O0 | texinfo_ginstall-info_O1 | 40 | 0.825 | 0.825 | 0.825 | 40 |
| texinfo_ginstall-info_O0 | texinfo_ginstall-info_O3 | 40 | 0.825 | 0.825 | 0.825 | 40 |
| groff_troff_O2 | groff_troff_O3 | 991 | 0.822 | 0.820 | 0.809 | 920 |
| coreutils4_printenv_O2 | coreutils4_printenv_O3 | 43 | 0.814 | 0.814 | 0.784 | 37 |
| groff_troff_O0 | groff_troff_O3 | 985 | 0.811 | 0.810 | 0.797 | 917 |
| which_which_O2 | which_which | 21 | 0.810 | 0.810 | 0.800 | 20 |
| coreutils_ptx_O2 | coreutils_ptx | 94 | 0.809 | 0.809 | 0.769 | 78 |
| groff_soelim_O1 | groff_soelim_O3 | 52 | 0.827 | 0.808 | 0.804 | 46 |
| inetutils_ping_O2 | inetutils_ping | 98 | 0.806 | 0.806 | 0.791 | 91 |
| coreutils4_uname_O2 | coreutils4_uname_O3 | 45 | 0.800 | 0.800 | 0.769 | 39 |
| time_time_O3 | time_time | 5 | 0.800 | 0.800 | 0.800 | 5 |
| coreutils_cat_O3 | coreutils_cat | 49 | 0.796 | 0.796 | 0.750 | 40 |
| groff_troff_O1 | groff_troff_O3 | 990 | 0.797 | 0.794 | 0.782 | 921 |
| coreutils4_users_O2 | coreutils4_users_O3 | 53 | 0.792 | 0.792 | 0.766 | 47 |
| coreutils_head_O2 | coreutils_head | 62 | 0.790 | 0.790 | 0.755 | 53 |
| coreutils_cp_O2 | coreutils_cp | 162 | 0.796 | 0.790 | 0.768 | 142 |
| coreutils_chmod_O2 | coreutils_chmod | 112 | 0.786 | 0.786 | 0.730 | 89 |
| coreutils_wc_O2 | coreutils_wc | 65 | 0.800 | 0.785 | 0.759 | 54 |
| coreutils_tsort_O2 | coreutils_tsort_O3 | 51 | 0.784 | 0.784 | 0.750 | 44 |
| texinfo_ginfo_O0 | texinfo_ginfo_O1 | 352 | 0.784 | 0.781 | 0.775 | 338 |
| texinfo_ginfo_O2 | texinfo_ginfo_O3 | 352 | 0.784 | 0.781 | 0.775 | 338 |
| coreutils_cat_O2 | coreutils_cat | 50 | 0.780 | 0.780 | 0.732 | 41 |
| texinfo_ginstall-info_O1 | texinfo_ginstall-info_O2 | 40 | 0.775 | 0.775 | 0.775 | 40 |
| texinfo_ginstall-info_O2 | texinfo_ginstall-info_O3 | 40 | 0.775 | 0.775 | 0.775 | 40 |
| coreutils_chown_O2 | coreutils_chown | 118 | 0.771 | 0.771 | 0.716 | 95 |
| coreutils_mv_O2 | coreutils_mv | 192 | 0.771 | 0.766 | 0.735 | 166 |
| less_lesskey_O2 | less_lesskey | 17 | 0.765 | 0.765 | 0.750 | 16 |
| coreutils3_tee_O2 | coreutils3_tee_O3 | 72 | 0.764 | 0.764 | 0.721 | 61 |
| coreutils_paste_O3 | coreutils_paste | 46 | 0.761 | 0.761 | 0.725 | 40 |
| groff_eqn_O2 | groff_eqn_O3 | 290 | 0.762 | 0.759 | 0.747 | 273 |
| patch_patch_O2 | patch_patch | 259 | 0.757 | 0.757 | 0.715 | 221 |
| groff_eqn_O1 | groff_eqn_O2 | 289 | 0.758 | 0.754 | 0.744 | 273 |
| jq_jq_O0 | jq_jq_O2 | 773 | 0.992 | 0.754 | 0.250 | 8 |
| coreutils_cp_O3 | coreutils_cp | 157 | 0.758 | 0.752 | 0.725 | 138 |
| coreutils4_sleep_O2 | coreutils4_sleep_O3 | 52 | 0.750 | 0.750 | 0.717 | 46 |
| psmisc_killall_O2 | psmisc_killall_O3 | 8 | 0.750 | 0.750 | 0.750 | 8 |
| coreutils3_link_O2 | coreutils3_link_O3 | 51 | 0.745 | 0.745 | 0.683 | 41 |
| diffutils2_cmp_O2 | diffutils2_cmp_O3 | 70 | 0.743 | 0.743 | 0.727 | 66 |
| coreutils3_comm_O2 | coreutils3_comm_O3 | 66 | 0.742 | 0.742 | 0.696 | 56 |
| coreutils4_who_O2 | coreutils4_who_O3 | 61 | 0.754 | 0.738 | 0.722 | 54 |
| libarchive_bsdtar_O2 | libarchive_bsdtar_O3 | 280 | 0.739 | 0.736 | 0.676 | 225 |
| coreutils_cat_O2 | coreutils_cat_O3 | 49 | 0.735 | 0.735 | 0.675 | 40 |
| libarchive_bsdcpio_O2 | libarchive_bsdcpio_O3 | 248 | 0.738 | 0.734 | 0.673 | 199 |
| m4_m4_O2 | m4_m4 | 454 | 0.736 | 0.731 | 0.706 | 408 |
| gnuchess_gnuchess_O0 | gnuchess_gnuchess_O2 | 532 | 0.733 | 0.731 | 0.707 | 484 |
| gnuchess_gnuchess_O1 | gnuchess_gnuchess_O3 | 532 | 0.733 | 0.731 | 0.707 | 484 |
| coreutils3_yes_O2 | coreutils3_yes_O3 | 48 | 0.729 | 0.729 | 0.690 | 42 |
| coreutils2_shred_O2 | coreutils2_shred | 92 | 0.728 | 0.728 | 0.684 | 79 |
| coreutils4_truncate_O2 | coreutils4_truncate_O3 | 51 | 0.725 | 0.725 | 0.674 | 43 |
| coreutils_tsort_O3 | coreutils_tsort | 51 | 0.725 | 0.725 | 0.682 | 44 |
| coreutils2_realpath_O2 | coreutils2_realpath | 80 | 0.725 | 0.725 | 0.672 | 67 |
| texinfo_ginfo_O0 | texinfo_ginfo_O3 | 352 | 0.730 | 0.724 | 0.719 | 338 |
| texinfo_ginfo_O1 | texinfo_ginfo_O2 | 352 | 0.730 | 0.724 | 0.719 | 338 |
| coreutils2_mktemp_O2 | coreutils2_mktemp_O3 | 50 | 0.720 | 0.720 | 0.674 | 43 |
| coreutils4_hostid_O2 | coreutils4_hostid_O3 | 46 | 0.717 | 0.717 | 0.675 | 40 |
| coreutils_paste_O2 | coreutils_paste_O3 | 46 | 0.717 | 0.717 | 0.675 | 40 |
| gettext_msgfmt_O2 | gettext_msgfmt_O3 | 53 | 0.736 | 0.717 | 0.667 | 42 |
| coreutils3_split_O2 | coreutils3_split_O3 | 88 | 0.716 | 0.716 | 0.684 | 79 |
| coreutils_tail_O2 | coreutils_tail | 109 | 0.716 | 0.716 | 0.663 | 92 |
| coreutils4_unlink_O2 | coreutils4_unlink_O3 | 49 | 0.714 | 0.714 | 0.659 | 41 |
| findutils_find_O2 | findutils_find | 354 | 0.715 | 0.712 | 0.680 | 316 |
| coreutils4_printf_O2 | coreutils4_printf_O3 | 69 | 0.725 | 0.710 | 0.672 | 58 |
| coreutils4_mkfifo_O2 | coreutils4_mkfifo_O3 | 51 | 0.706 | 0.706 | 0.651 | 43 |
| units_units_O3 | units_units | 129 | 0.705 | 0.705 | 0.698 | 126 |
| gnuchess_gnuchess_O2 | gnuchess_gnuchess_O3 | 532 | 0.707 | 0.705 | 0.678 | 484 |
| gperf_gperf_O0 | gperf_gperf | 105 | 0.733 | 0.705 | 0.678 | 87 |
| coreutils4_echo_O2 | coreutils4_echo_O3 | 44 | 0.705 | 0.705 | 0.658 | 38 |
| coreutils_ls_O2 | coreutils_ls | 186 | 0.704 | 0.704 | 0.660 | 162 |
| coreutils4_logname_O2 | coreutils4_logname_O3 | 47 | 0.702 | 0.702 | 0.659 | 41 |
| coreutils4_basename_O2 | coreutils4_basename_O3 | 50 | 0.700 | 0.700 | 0.659 | 44 |
| units_units_O2 | units_units | 130 | 0.700 | 0.700 | 0.693 | 127 |
| coreutils4_pwd_O2 | coreutils4_pwd_O3 | 53 | 0.698 | 0.698 | 0.644 | 45 |
| coreutils2_mktemp_O3 | coreutils2_mktemp | 49 | 0.673 | 0.694 | 0.651 | 43 |
| groff_eqn_O0 | groff_eqn_O1 | 284 | 0.694 | 0.694 | 0.677 | 269 |
| groff_eqn_O0 | groff_eqn_O3 | 284 | 0.694 | 0.694 | 0.677 | 269 |
| coreutils4_sha1sum_O2 | coreutils4_sha1sum_O3 | 55 | 0.691 | 0.691 | 0.653 | 49 |
| bison_bison_O2 | bison_bison | 857 | 0.694 | 0.691 | 0.650 | 749 |
| coreutils4_unexpand_O2 | coreutils4_unexpand_O3 | 58 | 0.690 | 0.690 | 0.654 | 52 |
| coreutils_head_O3 | coreutils_head | 61 | 0.689 | 0.689 | 0.635 | 52 |
| coreutils2_shred_O3 | coreutils2_shred | 91 | 0.681 | 0.681 | 0.637 | 80 |
| gnuchess_gnuchess_O0 | gnuchess_gnuchess_O3 | 532 | 0.680 | 0.679 | 0.649 | 484 |
| gnuchess_gnuchess_O1 | gnuchess_gnuchess_O2 | 532 | 0.680 | 0.679 | 0.649 | 484 |
| coreutils3_touch_O2 | coreutils3_touch_O3 | 93 | 0.688 | 0.677 | 0.642 | 81 |
| texinfo_ginstall-info_O0 | texinfo_ginstall-info_O2 | 40 | 0.675 | 0.675 | 0.675 | 40 |
| units_units_O2 | units_units_O3 | 129 | 0.674 | 0.674 | 0.667 | 126 |
| coreutils3_nproc_O2 | coreutils3_nproc_O3 | 49 | 0.673 | 0.673 | 0.628 | 43 |
| coreutils4_groups_O2 | coreutils4_groups_O3 | 49 | 0.673 | 0.673 | 0.628 | 43 |
| coreutils_mv_O2 | coreutils_mv_O3 | 180 | 0.678 | 0.672 | 0.628 | 156 |
| coreutils_uniq_O2 | coreutils_uniq_O3 | 64 | 0.672 | 0.672 | 0.604 | 53 |
| tar_tar_O2 | tar_tar | 746 | 0.672 | 0.672 | 0.635 | 672 |
| groff_tbl_O0 | groff_tbl_O1 | 194 | 0.675 | 0.670 | 0.623 | 167 |
| groff_tbl_O0 | groff_tbl_O3 | 194 | 0.675 | 0.670 | 0.623 | 167 |
| groff_tbl_O2 | groff_tbl_O3 | 200 | 0.680 | 0.670 | 0.626 | 171 |
| coreutils4_sha512sum_O2 | coreutils4_sha512sum_O3 | 57 | 0.667 | 0.667 | 0.627 | 51 |
| coreutils_nl_O3 | coreutils_nl | 51 | 0.686 | 0.667 | 0.628 | 43 |
| diffutils2_sdiff_O2 | diffutils2_sdiff_O3 | 81 | 0.667 | 0.667 | 0.649 | 77 |
| less_lessecho_O2 | less_lessecho_O3 | 3 | 0.667 | 0.667 | 0.667 | 3 |
| psmisc_prtstat_O2 | psmisc_prtstat_O3 | 3 | 0.667 | 0.667 | 0.667 | 3 |
| wdiff_wdiff_O2 | wdiff_wdiff_O3 | 24 | 0.667 | 0.667 | 0.667 | 24 |
| groff_tbl_O1 | groff_tbl_O2 | 199 | 0.673 | 0.663 | 0.620 | 171 |
| less_less_O3 | less_less | 402 | 0.662 | 0.662 | 0.636 | 371 |
| coreutils_sort_O2 | coreutils_sort | 171 | 0.661 | 0.661 | 0.618 | 152 |
| coreutils4_rmdir_O2 | coreutils4_rmdir_O3 | 56 | 0.679 | 0.661 | 0.617 | 47 |
| curl_curl_O2 | curl_curl_O3 | 1113 | 0.662 | 0.660 | 0.642 | 1050 |
| gperf_gperf_O0 | gperf_gperf_O1 | 105 | 0.695 | 0.657 | 0.632 | 87 |
| gperf_gperf_O0 | gperf_gperf_O2 | 105 | 0.695 | 0.657 | 0.632 | 87 |
| gperf_gperf_O0 | gperf_gperf_O3 | 105 | 0.695 | 0.657 | 0.632 | 87 |
| coreutils4_date_O2 | coreutils4_date_O3 | 96 | 0.677 | 0.656 | 0.627 | 83 |
| coreutils4_timeout_O2 | coreutils4_timeout_O3 | 55 | 0.655 | 0.655 | 0.612 | 49 |
| coreutils_fmt_O3 | coreutils_fmt | 55 | 0.655 | 0.655 | 0.596 | 47 |
| coreutils4_shuf_O2 | coreutils4_shuf_O3 | 92 | 0.652 | 0.652 | 0.600 | 80 |
| groff_pic_O2 | groff_pic_O3 | 261 | 0.655 | 0.651 | 0.622 | 238 |
| dos2unix_dos2unix_O2 | dos2unix_dos2unix_O3 | 40 | 0.650 | 0.650 | 0.650 | 40 |
| groff_pic_O0 | groff_pic_O1 | 255 | 0.647 | 0.647 | 0.615 | 234 |
| groff_pic_O0 | groff_pic_O3 | 255 | 0.647 | 0.647 | 0.615 | 234 |
| groff_pic_O1 | groff_pic_O2 | 260 | 0.650 | 0.646 | 0.618 | 238 |
| coreutils_cp_O2 | coreutils_cp_O3 | 157 | 0.650 | 0.643 | 0.601 | 138 |
| patch_patch_O2 | patch_patch_O3 | 242 | 0.645 | 0.640 | 0.583 | 206 |
| coreutils_cut_O2 | coreutils_cut_O3 | 50 | 0.640 | 0.640 | 0.591 | 44 |
| coreutils_tr_O2 | coreutils_tr_O3 | 61 | 0.639 | 0.639 | 0.569 | 51 |
| coreutils2_dd_O2 | coreutils2_dd | 102 | 0.647 | 0.637 | 0.591 | 88 |
| findutils_find_O2 | findutils_find_O3 | 325 | 0.637 | 0.637 | 0.595 | 291 |
| coreutils_fmt_O2 | coreutils_fmt_O3 | 55 | 0.636 | 0.636 | 0.574 | 47 |
| strace_strace_O2 | strace_strace_O3 | 1016 | 0.638 | 0.633 | 0.629 | 991 |
| coreutils3_id_O2 | coreutils3_id_O3 | 54 | 0.630 | 0.630 | 0.583 | 48 |
| hello_hello_O0 | hello_hello_O3 | 81 | 0.630 | 0.630 | 0.552 | 67 |
| hello_hello_O1 | hello_hello_O3 | 81 | 0.630 | 0.630 | 0.552 | 67 |
| hello_hello_O2 | hello_hello_O3 | 81 | 0.630 | 0.630 | 0.552 | 67 |
| coreutils_wc_O3 | coreutils_wc | 62 | 0.645 | 0.629 | 0.577 | 52 |
| patch_patch_O3 | patch_patch | 242 | 0.632 | 0.628 | 0.568 | 206 |
| coreutils3_fold_O2 | coreutils3_fold_O3 | 51 | 0.627 | 0.627 | 0.578 | 45 |
| texinfo_ginfo_O0 | texinfo_ginfo_O2 | 352 | 0.631 | 0.625 | 0.615 | 338 |
| texinfo_ginfo_O1 | texinfo_ginfo_O3 | 352 | 0.631 | 0.625 | 0.615 | 338 |
| coreutils_ls_O2 | coreutils_ls_O3 | 173 | 0.624 | 0.624 | 0.575 | 153 |
| coreutils_tr_O3 | coreutils_tr | 61 | 0.623 | 0.623 | 0.549 | 51 |
| groff_refer_O2 | groff_refer_O3 | 241 | 0.627 | 0.622 | 0.583 | 216 |
| coreutils4_numfmt_O2 | coreutils4_numfmt_O3 | 66 | 0.621 | 0.621 | 0.554 | 56 |
| groff_refer_O0 | groff_refer_O1 | 234 | 0.611 | 0.611 | 0.573 | 213 |
| groff_refer_O0 | groff_refer_O3 | 234 | 0.611 | 0.611 | 0.575 | 212 |
| which_which_O2 | which_which_O3 | 18 | 0.611 | 0.611 | 0.588 | 17 |
| coreutils_uniq_O3 | coreutils_uniq | 64 | 0.609 | 0.609 | 0.528 | 53 |
| coreutils3_stat_O2 | coreutils3_stat_O3 | 110 | 0.618 | 0.609 | 0.571 | 98 |
| coreutils_chown_O3 | coreutils_chown | 110 | 0.609 | 0.609 | 0.511 | 88 |
| coreutils_nl_O2 | coreutils_nl_O3 | 51 | 0.627 | 0.608 | 0.558 | 43 |
| groff_refer_O1 | groff_refer_O2 | 239 | 0.611 | 0.607 | 0.571 | 217 |
| coreutils_chmod_O3 | coreutils_chmod | 104 | 0.606 | 0.606 | 0.500 | 82 |
| coreutils_cut_O3 | coreutils_cut | 50 | 0.600 | 0.600 | 0.545 | 44 |
| psmisc_peekfd_O2 | psmisc_peekfd_O3 | 5 | 0.600 | 0.600 | 0.600 | 5 |
| time_time_O2 | time_time_O3 | 5 | 0.600 | 0.600 | 0.600 | 5 |
| time_time_O2 | time_time | 5 | 0.600 | 0.600 | 0.600 | 5 |
| gzip_gzip_O2 | gzip_gzip_O3 | 102 | 0.608 | 0.598 | 0.588 | 97 |
| coreutils2_shred_O2 | coreutils2_shred_O3 | 92 | 0.598 | 0.598 | 0.537 | 80 |
| groff_groff_O2 | groff_groff_O3 | 139 | 0.604 | 0.597 | 0.570 | 128 |
| coreutils3_env_O2 | coreutils3_env_O3 | 57 | 0.596 | 0.596 | 0.531 | 49 |
| flex_flex_O2 | flex_flex_O3 | 188 | 0.590 | 0.596 | 0.589 | 185 |
| strace_strace_O1 | strace_strace_O2 | 1016 | 0.600 | 0.594 | 0.590 | 991 |
| coreutils4_sum_O2 | coreutils4_sum_O3 | 69 | 0.594 | 0.594 | 0.525 | 59 |
| rcs_rcs_O2 | rcs_rcs_O3 | 270 | 0.600 | 0.593 | 0.585 | 258 |
| coreutils_sort_O2 | coreutils_sort_O3 | 157 | 0.592 | 0.592 | 0.546 | 141 |
| m4_m4_O3 | m4_m4 | 418 | 0.593 | 0.591 | 0.548 | 374 |
| coreutils4_readlink_O2 | coreutils4_readlink_O3 | 73 | 0.589 | 0.589 | 0.516 | 62 |
| coreutils_mv_O3 | coreutils_mv | 180 | 0.594 | 0.589 | 0.532 | 156 |
| gettext_xgettext_O2 | gettext_xgettext_O3 | 218 | 0.587 | 0.587 | 0.464 | 166 |
| binutils_objdump_O2 | binutils_objdump | 1907 | 0.585 | 0.581 | 0.561 | 1803 |
| coreutils_wc_O2 | coreutils_wc_O3 | 62 | 0.597 | 0.581 | 0.519 | 52 |
| groff_groff_O0 | groff_groff_O3 | 133 | 0.579 | 0.579 | 0.548 | 124 |
| inetutils2_dnsdomainname_O2 | inetutils2_dnsdomainname_O3 | 45 | 0.578 | 0.578 | 0.568 | 44 |
| coreutils_join_O2 | coreutils_join_O3 | 71 | 0.577 | 0.577 | 0.516 | 62 |
| coreutils_join_O3 | coreutils_join | 71 | 0.577 | 0.577 | 0.516 | 62 |
| coreutils2_dd_O3 | coreutils2_dd | 99 | 0.576 | 0.576 | 0.517 | 87 |
| less_less_O2 | less_less_O3 | 402 | 0.575 | 0.575 | 0.542 | 371 |
| coreutils4_tac_O2 | coreutils4_tac_O3 | 61 | 0.574 | 0.574 | 0.509 | 53 |
| coreutils_head_O2 | coreutils_head_O3 | 61 | 0.574 | 0.574 | 0.500 | 52 |
| diffutils2_diff3_O2 | diffutils2_diff3_O3 | 82 | 0.573 | 0.573 | 0.545 | 77 |
| groff_groff_O1 | groff_groff_O2 | 138 | 0.580 | 0.572 | 0.547 | 128 |
| binutils_size_O3 | binutils_size | 1083 | 0.574 | 0.572 | 0.555 | 1035 |
| groff_groff_O0 | groff_groff_O1 | 133 | 0.571 | 0.571 | 0.540 | 124 |
| less_lesskey_O2 | less_lesskey_O3 | 14 | 0.571 | 0.571 | 0.538 | 13 |
| less_lesskey_O3 | less_lesskey | 14 | 0.571 | 0.571 | 0.538 | 13 |
| plotutils_spline_O2 | plotutils_spline_O3 | 21 | 0.571 | 0.571 | 0.571 | 21 |
| plotutils_tek2plot_O2 | plotutils_tek2plot_O3 | 14 | 0.571 | 0.571 | 0.571 | 14 |
| coreutils_ptx_O2 | coreutils_ptx_O3 | 93 | 0.570 | 0.570 | 0.494 | 79 |
| gzip_gzip_O3 | gzip_gzip | 102 | 0.578 | 0.569 | 0.557 | 97 |
| coreutils_chmod_O2 | coreutils_chmod_O3 | 104 | 0.567 | 0.567 | 0.451 | 82 |
| binutils_size_O2 | binutils_size_O3 | 1083 | 0.570 | 0.567 | 0.550 | 1035 |
| binutils_addr2line_O2 | binutils_addr2line_O3 | 1080 | 0.569 | 0.566 | 0.548 | 1032 |
| coreutils_rm_O2 | coreutils_rm_O3 | 115 | 0.565 | 0.565 | 0.468 | 94 |
| coreutils_rm_O3 | coreutils_rm | 115 | 0.565 | 0.565 | 0.468 | 94 |
| coreutils2_df_O2 | coreutils2_df_O3 | 126 | 0.563 | 0.563 | 0.481 | 106 |
| plotutils_ode_O2 | plotutils_ode_O3 | 64 | 0.562 | 0.562 | 0.562 | 64 |
| cppi_cppi_O2 | cppi_cppi_O3 | 41 | 0.561 | 0.561 | 0.528 | 36 |
| which_which_O3 | which_which | 18 | 0.556 | 0.556 | 0.529 | 17 |
| gzip_gzip_O2 | gzip_gzip | 110 | 0.564 | 0.555 | 0.543 | 105 |
| binutils_addr2line_O2 | binutils_addr2line | 1212 | 0.559 | 0.554 | 0.538 | 1156 |
| coreutils2_stty_O3 | coreutils2_stty | 65 | 0.538 | 0.554 | 0.482 | 56 |
| coreutils2_du_O2 | coreutils2_du_O3 | 159 | 0.553 | 0.553 | 0.478 | 136 |
| flex_flex_O3 | flex_flex | 188 | 0.548 | 0.553 | 0.546 | 185 |
| coreutils2_realpath_O3 | coreutils2_realpath | 76 | 0.553 | 0.553 | 0.460 | 63 |
| enscript_enscript_O2 | enscript_enscript | 76 | 0.553 | 0.553 | 0.521 | 71 |
| binutils_strings_O2 | binutils_strings | 1219 | 0.557 | 0.552 | 0.536 | 1163 |
| coreutils2_ls_O3 | coreutils2_ls | 167 | 0.545 | 0.551 | 0.500 | 150 |
| coreutils2_ls_O2 | coreutils2_ls_O3 | 173 | 0.549 | 0.549 | 0.490 | 153 |
| coreutils_ls_O3 | coreutils_ls | 173 | 0.549 | 0.549 | 0.490 | 153 |
| binutils_nm-new_O2 | binutils_nm-new | 1234 | 0.553 | 0.549 | 0.531 | 1175 |
| coreutils2_df_O3 | coreutils2_df | 124 | 0.540 | 0.548 | 0.472 | 106 |
| coreutils2_realpath_O2 | coreutils2_realpath_O3 | 77 | 0.545 | 0.545 | 0.444 | 63 |
| coreutils2_stty_O2 | coreutils2_stty_O3 | 66 | 0.545 | 0.545 | 0.464 | 56 |
| plotutils_graph_O2 | plotutils_graph_O3 | 33 | 0.545 | 0.545 | 0.531 | 32 |
| cpio_cpio_O2 | cpio_cpio_O3 | 222 | 0.541 | 0.541 | 0.487 | 199 |
| cpio_cpio_O3 | cpio_cpio | 222 | 0.541 | 0.541 | 0.487 | 199 |
| binutils_nm-new_O2 | binutils_nm-new_O3 | 1099 | 0.542 | 0.539 | 0.520 | 1048 |
| binutils_strings_O2 | binutils_strings_O3 | 1085 | 0.541 | 0.537 | 0.520 | 1037 |
| coreutils3_mkdir_O2 | coreutils3_mkdir_O3 | 71 | 0.549 | 0.535 | 0.484 | 62 |
| coreutils2_du_O3 | coreutils2_du | 157 | 0.529 | 0.535 | 0.463 | 136 |
| coreutils3_seq_O2 | coreutils3_seq_O3 | 60 | 0.550 | 0.533 | 0.481 | 52 |
| bison_bison_O3 | bison_bison | 794 | 0.538 | 0.533 | 0.474 | 696 |
| coreutils3_expr_O2 | coreutils3_expr_O3 | 62 | 0.532 | 0.532 | 0.442 | 52 |
| tar_tar_O3 | tar_tar | 668 | 0.533 | 0.531 | 0.497 | 620 |
| inetutils2_logger_O2 | inetutils2_logger_O3 | 49 | 0.531 | 0.531 | 0.521 | 48 |
| plotutils_double_O2 | plotutils_double_O3 | 17 | 0.529 | 0.529 | 0.529 | 17 |
| coreutils_sort_O3 | coreutils_sort | 157 | 0.529 | 0.529 | 0.475 | 141 |
| groff_soelim_O2 | groff_soelim_O3 | 53 | 0.547 | 0.528 | 0.478 | 46 |
| direvent_direvent_O2 | direvent_direvent_O3 | 320 | 0.528 | 0.528 | 0.518 | 311 |
| coreutils_chown_O2 | coreutils_chown_O3 | 110 | 0.527 | 0.527 | 0.409 | 88 |
| coreutils_tail_O3 | coreutils_tail | 106 | 0.519 | 0.519 | 0.433 | 90 |
| nettle_sexp-conv_O2 | nettle_sexp-conv_O3 | 27 | 0.556 | 0.519 | 0.522 | 23 |
| findutils_find_O3 | findutils_find | 325 | 0.520 | 0.517 | 0.464 | 291 |
| binutils_objdump_O3 | binutils_objdump | 1715 | 0.519 | 0.515 | 0.493 | 1627 |
| coreutils2_sort_O2 | coreutils2_sort_O3 | 164 | 0.512 | 0.512 | 0.452 | 146 |
| coreutils4_pr_O2 | coreutils4_pr_O3 | 90 | 0.511 | 0.511 | 0.470 | 83 |
| screen_screen_O2 | screen_screen | 520 | 0.512 | 0.510 | 0.494 | 502 |
| make_make_O2 | make_make_O3 | 108 | 0.528 | 0.509 | 0.495 | 101 |
| make_make_O3 | make_make | 108 | 0.528 | 0.509 | 0.495 | 101 |
| coreutils4_factor_O2 | coreutils4_factor_O3 | 73 | 0.507 | 0.507 | 0.455 | 66 |
| enscript_enscript_O3 | enscript_enscript | 75 | 0.507 | 0.507 | 0.471 | 70 |
| coreutils_ptx_O3 | coreutils_ptx | 93 | 0.505 | 0.505 | 0.418 | 79 |
| acct_dump-utmp_O2 | acct_dump-utmp_O3 | 24 | 0.583 | 0.500 | 0.500 | 20 |
| gettext_msgmerge_O2 | gettext_msgmerge_O3 | 12 | 0.500 | 0.500 | 0.400 | 10 |
| groff_soelim_O1 | groff_soelim_O2 | 52 | 0.519 | 0.500 | 0.457 | 46 |

## B6 — Non-`sub_` graphs

- match_index keys pointing to non-`_sub_` graph files: **42** (all 42 listed in audit.json → B6.match_index_non_sub_keys; e.g. `bash_bash_O0_memset.json` → real_name `_rl_read_file`, `binutils_addr2line_O1_strchr.json` → `bfd_cache_size`: BAP mis-named these subs with a library symbol and the matcher matched them by address to an unrelated label — 42 label-noise samples).
- Non-`sub_` graph files on disk: **100,843** (6,267 distinct names). Top 20: main (921), __stack_chk_fail (916), free (898), exit (896), strlen (889), strcmp (883), fwrite (874), malloc (871), __errno_location (854), memcpy (844), fclose (825), __cxa_finalize (818), realloc (810), strncmp (799), memset (787), __ctype_b_loc (756), calloc (749), abort (746), memcmp (733), fflush (712).
- Could the loader load them? **Only the 42 that are match_index keys.** The loader never globs `data/graphs`; it iterates `for graph_path, match_info in self.match_index.items():` (src/preprocessing/build_dataset.py:228 `for graph_path, match_info in self.match_index.items():` (+229 `if not os.path.exists(graph_path): continue`); dev L215 identical) and opens only those paths. Non-sub graphs are otherwise touched only by thunk resolution (`<binary>_<internal_callees[0]>.json`, unified L249, dev L236) — which can pull in a PLT-stub graph such as `<binary>_free.json` if a thunk's first callee is a stub — and never as samples.

## B8 — Split integrity

- split file sizes: {'train': 431, 'val': 17, 'test': 18, 'excluded': 4}. match_index binaries: 881. **414 match_index binaries (98,331 functions = 32.6% of match_index) are in NO split list → the loader silently adds them to train** (get_splits(): new_binaries = all_binaries - known; train_bins_set.update(new_binaries) (unified L671-673, dev L558-562)). This includes ALL binaries of the clean-7 packages dash, gettext, psmisc, and all O1/O3 builds of packages whose O0/O2 are split, and all of coreutils3/coreutils4/binutils2/busybox/plotutils/etc.
- Split-file binaries absent from match_index (contribute 0 samples): ['grep_grep', 'sed_sed', 'wget_wget_O0'].

### Unsplit (default→train) binaries by package

| package | #bins | binaries |
|---|---|---|
| acct | 12 | acct_ac_O1, acct_ac_O3, acct_accton_O1, acct_accton_O3, acct_dump-utmp_O1, acct_dump-utmp_O3, acct_last_O1, acct_last_O3, acct_lastcomm_O1, acct_lastcomm_O3, acct_sa_O1, acct_sa_O3 |
| bc | 4 | bc_bc_O1, bc_bc_O3, bc_dc_O1, bc_dc_O3 |
| binutils2 | 16 | binutils2_ar_O0, binutils2_ar_O1, binutils2_ar_O2, binutils2_ar_O3, binutils2_objcopy_O0, binutils2_objcopy_O1, binutils2_objcopy_O2, binutils2_objcopy_O3, binutils2_ranlib_O0, binutils2_ranlib_O1, binutils2_ranlib_O2, binutils2_ranlib_O3, binutils2_strip-new_O0, binutils2_strip-new_O1, binutils2_strip-new_O2, binutils2_strip-new_O3 |
| busybox | 4 | busybox_busybox_O0, busybox_busybox_O1, busybox_busybox_O2, busybox_busybox_O3 |
| bzip2 | 2 | bzip2_bzip2_O1, bzip2_bzip2_O3 |
| coreutils3 | 68 | coreutils3_comm_O0, coreutils3_comm_O1, coreutils3_comm_O2, coreutils3_comm_O3, coreutils3_dirname_O0, coreutils3_dirname_O1, coreutils3_dirname_O2, coreutils3_dirname_O3, coreutils3_env_O0, coreutils3_env_O1, coreutils3_env_O2, coreutils3_env_O3, coreutils3_expand_O0, coreutils3_expand_O1, coreutils3_expand_O2, coreutils3_expand_O3, coreutils3_expr_O0, coreutils3_expr_O1, coreutils3_expr_O2, coreutils3_expr_O3, coreutils3_fold_O0, coreutils3_fold_O1, coreutils3_fold_O2, coreutils3_fold_O3, coreutils3_id_O0, coreutils3_id_O1, coreutils3_id_O2, coreutils3_id_O3, coreutils3_link_O0, coreutils3_link_O1, coreutils3_link_O2, coreutils3_link_O3, coreutils3_ln_O0, coreutils3_ln_O1, coreutils3_ln_O2, coreutils3_ln_O3, coreutils3_mkdir_O0, coreutils3_mkdir_O1, coreutils3_mkdir_O2, coreutils3_mkdir_O3, coreutils3_nproc_O0, coreutils3_nproc_O1, coreutils3_nproc_O2, coreutils3_nproc_O3, coreutils3_seq_O0, coreutils3_seq_O1, coreutils3_seq_O2, coreutils3_seq_O3, coreutils3_split_O0, coreutils3_split_O1, coreutils3_split_O2, coreutils3_split_O3, coreutils3_stat_O0, coreutils3_stat_O1, coreutils3_stat_O2, coreutils3_stat_O3, coreutils3_tee_O0, coreutils3_tee_O1, coreutils3_tee_O2, coreutils3_tee_O3, coreutils3_touch_O0, coreutils3_touch_O1, coreutils3_touch_O2, coreutils3_touch_O3, coreutils3_yes_O0, coreutils3_yes_O1, coreutils3_yes_O2, coreutils3_yes_O3 |
| coreutils4 | 124 | coreutils4_basename_O0, coreutils4_basename_O1, coreutils4_basename_O2, coreutils4_basename_O3, coreutils4_date_O0, coreutils4_date_O1, coreutils4_date_O2, coreutils4_date_O3, coreutils4_echo_O0, coreutils4_echo_O1, coreutils4_echo_O2, coreutils4_echo_O3, coreutils4_factor_O0, coreutils4_factor_O1, coreutils4_factor_O2, coreutils4_factor_O3, coreutils4_groups_O0, coreutils4_groups_O1, coreutils4_groups_O2, coreutils4_groups_O3, coreutils4_hostid_O0, coreutils4_hostid_O1, coreutils4_hostid_O2, coreutils4_hostid_O3, coreutils4_logname_O0, coreutils4_logname_O1, coreutils4_logname_O2, coreutils4_logname_O3, coreutils4_mkfifo_O0, coreutils4_mkfifo_O1, coreutils4_mkfifo_O2, coreutils4_mkfifo_O3, coreutils4_numfmt_O0, coreutils4_numfmt_O1, coreutils4_numfmt_O2, coreutils4_numfmt_O3, coreutils4_pr_O0, coreutils4_pr_O1, coreutils4_pr_O2, coreutils4_pr_O3, coreutils4_printenv_O0, coreutils4_printenv_O1, coreutils4_printenv_O2, coreutils4_printenv_O3, coreutils4_printf_O0, coreutils4_printf_O1, coreutils4_printf_O2, coreutils4_printf_O3, coreutils4_pwd_O0, coreutils4_pwd_O1, coreutils4_pwd_O2, coreutils4_pwd_O3, coreutils4_readlink_O0, coreutils4_readlink_O1, coreutils4_readlink_O2, coreutils4_readlink_O3, coreutils4_rmdir_O0, coreutils4_rmdir_O1, coreutils4_rmdir_O2, coreutils4_rmdir_O3, coreutils4_sha1sum_O0, coreutils4_sha1sum_O1, coreutils4_sha1sum_O2, coreutils4_sha1sum_O3, coreutils4_sha512sum_O0, coreutils4_sha512sum_O1, coreutils4_sha512sum_O2, coreutils4_sha512sum_O3, coreutils4_shuf_O0, coreutils4_shuf_O1, coreutils4_shuf_O2, coreutils4_shuf_O3, coreutils4_sleep_O0, coreutils4_sleep_O1, coreutils4_sleep_O2, coreutils4_sleep_O3, coreutils4_sum_O0, coreutils4_sum_O1, coreutils4_sum_O2, coreutils4_sum_O3, coreutils4_tac_O0, coreutils4_tac_O1, coreutils4_tac_O2, coreutils4_tac_O3, coreutils4_timeout_O0, coreutils4_timeout_O1, coreutils4_timeout_O2, coreutils4_timeout_O3, coreutils4_truncate_O0, coreutils4_truncate_O1, coreutils4_truncate_O2, coreutils4_truncate_O3, coreutils4_tty_O0, coreutils4_tty_O1, coreutils4_tty_O2, coreutils4_tty_O3, coreutils4_uname_O0, coreutils4_uname_O1, coreutils4_uname_O2, coreutils4_uname_O3, coreutils4_unexpand_O0, coreutils4_unexpand_O1, coreutils4_unexpand_O2, coreutils4_unexpand_O3, coreutils4_unlink_O0, coreutils4_unlink_O1, coreutils4_unlink_O2, coreutils4_unlink_O3, coreutils4_uptime_O0, coreutils4_uptime_O1, coreutils4_uptime_O2, coreutils4_uptime_O3, coreutils4_users_O0, coreutils4_users_O1, coreutils4_users_O2, coreutils4_users_O3, coreutils4_who_O0, coreutils4_who_O1, coreutils4_who_O2, coreutils4_who_O3, coreutils4_whoami_O0, coreutils4_whoami_O1, coreutils4_whoami_O2, coreutils4_whoami_O3 |
| curl | 2 | curl_curl_O1, curl_curl_O3 |
| dash | 4 | dash_dash_O0, dash_dash_O1, dash_dash_O2, dash_dash_O3 |
| diffutils2 | 16 | diffutils2_cmp_O0, diffutils2_cmp_O1, diffutils2_cmp_O2, diffutils2_cmp_O3, diffutils2_diff3_O0, diffutils2_diff3_O1, diffutils2_diff3_O2, diffutils2_diff3_O3, diffutils2_diff_O0, diffutils2_diff_O1, diffutils2_diff_O2, diffutils2_diff_O3, diffutils2_sdiff_O0, diffutils2_sdiff_O1, diffutils2_sdiff_O2, diffutils2_sdiff_O3 |
| dos2unix | 4 | dos2unix_dos2unix_O1, dos2unix_dos2unix_O3, dos2unix_unix2dos_O1, dos2unix_unix2dos_O3 |
| dropbear | 6 | dropbear_dbclient_O0, dropbear_dbclient_O2, dropbear_dropbear_O0, dropbear_dropbear_O2, dropbear_dropbearkey_O0, dropbear_dropbearkey_O2 |
| ed | 4 | ed_ed_O0, ed_ed_O1, ed_ed_O2, ed_ed_O3 |
| expat | 2 | expat_xmlwf_O1, expat_xmlwf_O3 |
| gettext | 12 | gettext_msgfmt_O0, gettext_msgfmt_O1, gettext_msgfmt_O2, gettext_msgfmt_O3, gettext_msgmerge_O0, gettext_msgmerge_O1, gettext_msgmerge_O2, gettext_msgmerge_O3, gettext_xgettext_O0, gettext_xgettext_O1, gettext_xgettext_O2, gettext_xgettext_O3 |
| gnuchess | 4 | gnuchess_gnuchess_O0, gnuchess_gnuchess_O1, gnuchess_gnuchess_O2, gnuchess_gnuchess_O3 |
| htop | 2 | htop_htop_O1, htop_htop_O3 |
| indent | 2 | indent_indent_O1, indent_indent_O3 |
| inetutils2 | 12 | inetutils2_dnsdomainname_O0, inetutils2_dnsdomainname_O1, inetutils2_dnsdomainname_O2, inetutils2_dnsdomainname_O3, inetutils2_hostname_O0, inetutils2_hostname_O1, inetutils2_hostname_O2, inetutils2_hostname_O3, inetutils2_logger_O0, inetutils2_logger_O1, inetutils2_logger_O2, inetutils2_logger_O3 |
| jq | 2 | jq_jq_O1, jq_jq_O3 |
| libarchive | 6 | libarchive_bsdcat_O1, libarchive_bsdcat_O3, libarchive_bsdcpio_O1, libarchive_bsdcpio_O3, libarchive_bsdtar_O1, libarchive_bsdtar_O3 |
| libpng | 2 | libpng_pngtest_O1, libpng_pngtest_O3 |
| libxml2 | 2 | libxml2_xmllint_O1, libxml2_xmllint_O3 |
| libyaml | 4 | libyaml_run-emitter_O1, libyaml_run-emitter_O3, libyaml_run-parser_O1, libyaml_run-parser_O3 |
| lua | 4 | lua_lua_O1, lua_lua_O3, lua_luac_O1, lua_luac_O3 |
| lz4 | 2 | lz4_lz4_O1, lz4_lz4_O3 |
| mailutils | 8 | mailutils_frm_O0, mailutils_frm_O2, mailutils_mail_O0, mailutils_mail_O2, mailutils_messages_O0, mailutils_messages_O2, mailutils_readmsg_O0, mailutils_readmsg_O2 |
| nettle | 8 | nettle_nettle-hash_O0, nettle_nettle-hash_O1, nettle_nettle-hash_O2, nettle_nettle-hash_O3, nettle_sexp-conv_O0, nettle_sexp-conv_O1, nettle_sexp-conv_O2, nettle_sexp-conv_O3 |
| nginx | 2 | nginx_nginx_O0, nginx_nginx_O2 |
| pcre2 | 4 | pcre2_pcre2grep_O1, pcre2_pcre2grep_O3, pcre2_pcre2test_O1, pcre2_pcre2test_O3 |
| pigz | 2 | pigz_pigz_O0, pigz_pigz_O2 |
| plotutils | 28 | plotutils_double_O0, plotutils_double_O1, plotutils_double_O2, plotutils_double_O3, plotutils_graph_O0, plotutils_graph_O1, plotutils_graph_O2, plotutils_graph_O3, plotutils_ode_O0, plotutils_ode_O1, plotutils_ode_O2, plotutils_ode_O3, plotutils_plot_O0, plotutils_plot_O1, plotutils_plot_O2, plotutils_plot_O3, plotutils_plotfont_O0, plotutils_plotfont_O1, plotutils_plotfont_O2, plotutils_plotfont_O3, plotutils_spline_O0, plotutils_spline_O1, plotutils_spline_O2, plotutils_spline_O3, plotutils_tek2plot_O0, plotutils_tek2plot_O1, plotutils_tek2plot_O2, plotutils_tek2plot_O3 |
| psmisc | 16 | psmisc_killall_O0, psmisc_killall_O1, psmisc_killall_O2, psmisc_killall_O3, psmisc_peekfd_O0, psmisc_peekfd_O1, psmisc_peekfd_O2, psmisc_peekfd_O3, psmisc_prtstat_O0, psmisc_prtstat_O1, psmisc_prtstat_O2, psmisc_prtstat_O3, psmisc_pstree_O0, psmisc_pstree_O1, psmisc_pstree_O2, psmisc_pstree_O3 |
| rcs | 2 | rcs_rcs_O1, rcs_rcs_O3 |
| rsync | 2 | rsync_rsync_O0, rsync_rsync_O2 |
| rush | 2 | rush_rush_O1, rush_rush_O3 |
| socat | 2 | socat_socat_O0, socat_socat_O2 |
| sqlite | 2 | sqlite_sqlite3_O1, sqlite_sqlite3_O3 |
| strace | 2 | strace_strace_O1, strace_strace_O3 |
| tmux | 2 | tmux_tmux_O0, tmux_tmux_O2 |
| tree | 2 | tree_tree_O1, tree_tree_O3 |
| wdiff | 4 | wdiff_wdiff_O0, wdiff_wdiff_O1, wdiff_wdiff_O2, wdiff_wdiff_O3 |
| zlib | 4 | zlib_example_O1, zlib_example_O3, zlib_minigzip_O1, zlib_minigzip_O3 |

### Membership of the sensitive packages

| package | train | val | test | excluded | in match_index | in labels dir | unsplit→train |
|---|---|---|---|---|---|---|---|
| dash | — | — | — | — | dash_dash_O0, dash_dash_O1, dash_dash_O2, dash_dash_O3 | dash_dash_O0, dash_dash_O1, dash_dash_O2, dash_dash_O3 | dash_dash_O0, dash_dash_O1, dash_dash_O2, dash_dash_O3 |
| gettext | — | — | — | — | 12 bins | 12 bins | 12 bins |
| psmisc | — | — | — | — | 16 bins | 16 bins | 16 bins |
| recutils | — | — | — | — | — | 36 bins | — |
| nginx118 | — | — | — | — | — | — | — |
| angie | — | — | — | — | — | — | — |
| tengine | — | — | — | — | — | tengine_nginx_O0, tengine_nginx_O2 | — |
| grep | grep_grep, grep_grep_O0, grep_grep_O1, grep_grep_O2, grep_grep_O3 | — | — | — | grep_grep_O0, grep_grep_O1, grep_grep_O2, grep_grep_O3 | grep_grep_O0, grep_grep_O1, grep_grep_O2, grep_grep_O3 | — |
| sed | sed_sed, sed_sed_O0, sed_sed_O1, sed_sed_O2, sed_sed_O3 | — | — | — | sed_sed_O0, sed_sed_O1, sed_sed_O2, sed_sed_O3 | sed_sed_O0, sed_sed_O1, sed_sed_O2, sed_sed_O3 | — |

Notes: recutils has labels+graphs for 36 binaries but 0 match_index entries and is in no split list (so it is clean only by accident of never being matched); nginx118/angie have external.json files but no graphs/labels/match entries at all; tengine has labels+graphs, 0 matches, no split entry; grep/sed are explicitly train (incl. non-existent `grep_grep`/`sed_sed` ids). `nginx_nginx_O0/O2` (a different nginx build) IS in match_index (2,930 sub_ graphs, 0.57 channel) and unsplit → train.

## Files
- audit.json — all numbers (per-binary tables under B2.per_binary, B3.per_binary, B5.all_pairs).
- scan.tsv — per-graph raw scan (488,325 rows).
- scan_graphs.py / analyze.py — scripts.
