# RARC Stage 0 — Retrieval Instrumentation Audit

- frozen U0 n: 13581
- exported queries: 13581 (missing 0, duplicate 0)
- **exported final rank-1 == frozen U0_name: 13581/13581 = 100.0000%**
- BinFilter: fingerprints for 936 binaries; active on 8/77 query binaries; per-candidate `binfilter_score` not defined in production (binary Jaccard mask only) -> column null, `binfilter_pass` exported
- top-20 coverage: mean 20.00 rows/query (queries with <20 rows passing filter: 0)
- index: paper-clean 241,174 from ztr_full_control.npz; queries: zq_clean7_fresh.npz (predict path)
- replicated code: `final_composition.knn_binfilter` (cosine + Jaccard>=0.5 fingerprint mask, inert fallback, argmax)
- tie handling: agreement requires the frozen name inside the EXACT max-cosine tie set (tol 1e-5); export rank-1 = the anchor's realized tie choice; remaining ranks ordered (filtered cos desc, pre-filter rank asc). Ties arise from duplicate functions across optimization levels with identical embeddings.
- real mismatches (frozen name NOT in max-tie set): 0 

## Decision: **PASS**
