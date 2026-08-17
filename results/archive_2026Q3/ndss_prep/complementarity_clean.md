# Clean-checkpoint complementarity (strict split, baseline preds)

## Regime table (7-pkg xproj)

| Regime | N | share | Dec F1 | Dec EM | kNN F1 | kNN EM | Gate F1 | Gate EM |
|---|---|---|---|---|---|---|---|---|
| seen | 10133 | 74.6% | 0.677 | 55.4% | 0.737 | 62.6% | 0.703 | 58.6% |
| novel | 1539 | 11.3% | 0.076 | 0.0% | 0.095 | 0.0% | 0.080 | 0.0% |
| oov | 1909 | 14.1% | 0.081 | 0.0% | 0.075 | 0.0% | 0.077 | 0.0% |

## NCT/FT x regime cross-tab (row %)

| Partition | seen | novel | oov |
|---|---|---|---|
| NCT (n=7917) | 93.8% | 4.2% | 2.0% |
| FT (n=5664) | 47.8% | 21.3% | 30.9% |

## Per-package regime shares (%)

| pkg | seen | novel | oov |
|---|---|---|---|
| angie | 90.3% | 6.2% | 3.4% |
| nginx118 | 99.7% | 0.3% | 0.0% |
| tengine | 81.4% | 13.9% | 4.7% |
| dash | 81.9% | 6.1% | 12.0% |
| gettext | 4.7% | 56.0% | 39.3% |
| psmisc | 51.8% | 41.5% | 6.6% |
| recutils | 55.3% | 6.4% | 38.3% |
