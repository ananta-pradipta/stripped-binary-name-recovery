# Leakage Audit — 7-pkg Cross-Project Set vs Train Definitions

- xproj set: 13581 functions from $WORKSPACE/results/hybrid_xfl_REPRO.json (expected ~13581, deviation 0.00%)
- Train A (headline `best_model_pdec_xfl.pt`): reconstructed 279299 fns (logged 243805) — dash/gettext/psmisc IN train
- Train B (paper-clean `best_model_paper_clean.pt`): reconstructed 274106 fns (logged 241174) — all 7 xproj pkgs (+grep/sed) excluded
- Sub-token splitter: build_votes (underscore + camelCase, lowercased); freq1 = seen at least once in train, mc2 = Votes min_count>=2 vocab rule.
- Caveat: train sets reconstructed by binary membership from the Apr 22 match-index snapshot; loader-level function filtering (~12%) not replicated.

## Definition A (headline pdec_xfl — leaky)
Train: 44559 unique names, 16523 sub-token types (freq>=1), 15505 (min_count>=2)

| Package | n | Verbatim name % | Type cov % (f1/mc2) | Token cov % (f1/mc2) | Fully composable % (f1/mc2) |
|---|---|---|---|---|---|
| nginx118 | 3470 | 99.7 | 99.5 / 92.2 | 100.0 / 99.7 | 99.9 / 98.4 |
| angie | 3893 | 90.3 | 97.9 / 91.1 | 99.2 / 98.9 | 96.3 / 94.9 |
| tengine | 554 | 81.4 | 95.7 / 87.7 | 98.6 / 96.9 | 95.1 / 88.4 |
| recutils | 2550 | 55.2 | 95.3 / 95.3 | 85.2 / 85.2 | 61.3 / 61.3 |
| dash | 1324 | 100.0 | 100.0 / 97.0 | 100.0 / 98.8 | 100.0 / 98.6 |
| gettext | 1518 | 100.0 | 100.0 / 96.3 | 100.0 / 99.5 | 100.0 / 98.5 |
| psmisc | 272 | 100.0 | 100.0 / 90.2 | 100.0 / 97.6 | 100.0 / 94.9 |
| ALL | 13581 | 88.0 | 97.1 / 91.5 | 97.7 / 97.3 | 91.4 / 90.0 |

Verbatim % uses normalized names (metrics.normalize_name); raw-string match is in the JSON.

## Definition B (paper-clean)
Train: 43738 unique names, 16271 sub-token types (freq>=1), 15227 (min_count>=2)

| Package | n | Verbatim name % | Type cov % (f1/mc2) | Token cov % (f1/mc2) | Fully composable % (f1/mc2) |
|---|---|---|---|---|---|
| nginx118 | 3470 | 99.7 | 99.5 / 92.2 | 100.0 / 99.7 | 99.9 / 98.4 |
| angie | 3893 | 90.3 | 97.9 / 91.1 | 99.2 / 98.9 | 96.3 / 94.9 |
| tengine | 554 | 81.4 | 95.7 / 87.7 | 98.6 / 96.9 | 95.1 / 88.4 |
| recutils | 2550 | 55.2 | 95.3 / 95.3 | 85.2 / 85.2 | 61.3 / 61.3 |
| dash | 1324 | 81.9 | 87.6 / 70.0 | 89.0 / 75.5 | 87.5 / 72.3 |
| gettext | 1518 | 4.2 | 67.4 / 66.8 | 77.0 / 76.9 | 44.5 / 44.3 |
| psmisc | 272 | 51.8 | 86.3 / 83.3 | 95.3 / 94.5 | 89.7 / 87.9 |
| ALL | 13581 | 74.5 | 85.2 / 77.4 | 95.2 / 94.5 | 83.8 / 81.2 |

Verbatim % uses normalized names (metrics.normalize_name); raw-string match is in the JSON.

