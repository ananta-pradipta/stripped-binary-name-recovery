# H2 Qualitative Examples (verbatim from `phase2_full_dump.tsv`, seed 42)

All predictions are actual system outputs; nothing is cleaned or invented.
H2 scores are the validation-calibrated `Platt(sim)×Platt(exist)` values.
Recall the operating guidance from the frontier: scores ≥~0.4 sit in the
≥70%-precision regime; scores below ~0.1 are below any useful operating point.

## A. Retrieval correct + composition corroborates

**`quotearg_buffer_restyled`** (recutils, SEEN)
- retrieval: `quotearg_buffer_restyled` ✓ (F1 1.0)
- H2 tail hints: `restyled: 0.566`
- Reading: the highest-confidence H2 hint corroborates the retrieved identifier's
  rarest component (`restyled` is a tail atom, training rank >4000). Agreement
  between independent heads raises analyst trust.

## B. Retrieval partially wrong + composition adds the missing concept

**`setlocale_null_androidfix`** (recutils, SEEN)
- retrieval: `setlocale_null_unlocked` (F1 0.667 — right family, wrong suffix)
- H2 tail hints: `androidfix: 0.359`
- Reading: the flagship pattern. Retrieval lands on the correct gnulib family
  but the wrong variant; H2 supplies exactly the missing tail atom
  (`androidfix`), which also tells the analyst *which platform quirk* this
  variant handles. This is composition adding information retrieval
  structurally cannot (the `androidfix` variant is not the nearest neighbor).

## C. Retrieval F1 = 0 + H2 recovers a useful concept

**`deregister_tm_clones`** (angie, retrieval F1 = 0)
- retrieval: `ngx_test_full_name` (completely wrong)
- H2 tail hints: `clones: 0.420`
- Reading: retrieval is misled (runtime scaffolding embedded in an nginx-family
  binary); H2's calibrated hint `clones` at 0.42 is enough to steer an analyst
  toward the transactional-memory clone-registration idiom instead of the
  bogus nginx name.

## D. Rare/very-rare successful hints (high confidence)

The three atoms above — `restyled` (0.566), `clones` (0.420),
`androidfix` (0.359) — are all tail atoms (training rank >4000, counts ≤ a
handful). At these scores they sit in the ≥70%-precision region of the H2
frontier; under H1's raw-cosine channel the same atoms surface only below its
50%-precision operating point. This is the frontier improvement shown as
individual predictions.

## E. Appropriate abstention

**`msgfmt_desktop_bulk`** (gettext, PARTIAL_OOV, retrieval F1 = 0)
- retrieval: `Curl_sha256it` (wrong)
- H2 tail hints: `pgnreplayv: 0.002` (top score ≈ zero)
- Reading: the ground-truth tail atoms (`msgfmt`, `desktop`, `bulk`) are not
  recoverable from z, and H2's calibration correctly collapses its confidence
  to ~0 — the analyst sees *no* tail hints at any reasonable threshold, rather
  than confident noise. Silence is the correct output here.

## F. Honest failure — confidently wrong

**`stputs`** (dash, FULL_OOV, retrieval F1 = 0)
- retrieval: `xstrdup` (wrong)
- H2 tail hints: `xstrdup: 0.819` (highest-confidence hint — and wrong)
- Reading: the worst case. Both heads read this dash string-buffer helper as
  `xstrdup`, and H2 asserts it at 0.82 — above every precision band. The
  failure is informative: H2's confidence is a property of z, so when the
  *representation* confuses two allocation/copy idioms, both heads fail
  coherently and calibration cannot save it. Roughly 1 in 20 hints at this
  confidence level are of this kind (frontier: ~95% precision at 0.9%
  coverage) — the false-hint risk the paper must state.
