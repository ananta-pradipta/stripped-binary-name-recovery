# Publication Venues for Binary Function Name Recovery

---

## Slide 1: All Relevant Venues

### Top-Tier Security (A*/A)
| Venue | Type | Deadline (typical) | Relevance | Published Work |
|---|---|---|---|---|
| **USENIX Security** | Conf | Feb/Jun/Oct (3 cycles) | Binary analysis track | BLens (2025), DIRTY (2022) |
| **IEEE S&P (Oakland)** | Conf | Jun/Dec (2 cycles) | ML for security | XFL (2023), CALLEE (2023) |
| **ACM CCS** | Conf | Jan/May (2 cycles) | Program analysis | SymLM (2022) |
| **NDSS** | Conf | Apr/Jul (2 cycles) | Applied security | SYMGEN (2025), GenNm (2025) |

### Top-Tier SE/PL (A*/A)
| Venue | Type | Deadline | Relevance | Published Work |
|---|---|---|---|---|
| **ACM OOPSLA** | Conf | Apr/Oct | PL + ML | NERO (2020) |
| **ACM FSE** | Conf | Sep | Software engineering | Epitome (2024) |
| **ACM ISSTA** | Conf | Jan | Testing/analysis | NFRE (2021) |
| **ACM TOSEM** | Journal | Rolling | SE journal (top-tier) | — |
| **IEEE TSE** | Journal | Rolling | SE journal (top-tier) | — |
| **ICSE** | Conf | Sep | Flagship SE conf | — |

### Top-Tier AI/ML
| Venue | Type | Deadline | Relevance |
|---|---|---|---|
| **NeurIPS** | Conf | May | ML + applications |
| **ICML** | Conf | Jan | ML methodology |
| **AAAI** | Conf | Aug | AI applications |
| **ICLR** | Conf | Sep | Representation learning |

### Mid-Tier / Specialized (B+)
| Venue | Type | Deadline | Relevance |
|---|---|---|---|
| **AsiaCCS** | Conf | Oct | Security (Asia) — AsmDepictor (2023) |
| **ACSAC** | Conf | Jun | Applied security |
| **RAID** | Conf | Mar | Intrusion/malware detection |
| **SecureComm** | Conf | Jun | Security + networking |
| **ACM ASIACCS** | Conf | Oct | Applied crypto/security |
| **IEEE SANER** | Conf | Oct | Software analysis + RE |
| **MSR** | Conf | Nov | Mining software repos |
| **ASE** | Conf | Apr | Automated SE |
| **ESEC/FSE** | Conf | varies | Same as FSE |
| **ESORICS** | Conf | Jan | European security |
| **IEEE TDSC** | Journal | Rolling | Dependable/secure computing |

---

## Slide 2: Curated Recommendations

### Tier 1 -- Best Fit (security + binary analysis audience)

| # | Venue | Why | Fit Score |
|---|---|---|---|
| 1 | **USENIX Security** | Primary binary analysis venue. BLens published here. Strong systems track. | 10/10 |
| 2 | **NDSS** | 2 papers on function naming in 2025 alone (SYMGEN, GenNm). Active community. | 10/10 |
| 3 | **IEEE S&P** | XFL published here. ML-for-security is a growing track. | 9/10 |
| 4 | **ACM CCS** | SymLM published here. Strong applied ML track. | 9/10 |

### Tier 2 -- Strong Fit (SE/PL audience, values methodology)

| # | Venue | Why | Fit Score |
|---|---|---|---|
| 5 | **ISSTA** | NFRE published here. Values empirical evaluation + ablation studies. | 8/10 |
| 6 | **FSE** | Epitome published here. Good for novel tokenization + fusion contributions. | 8/10 |
| 7 | **ICSE** | Flagship SE. Would need strong SE angle (tool, developer study). | 7/10 |
| 8 | **ASE** | Automated SE. Good for the pipeline/architecture contribution. | 7/10 |

### Tier 3 -- Backup / Workshop Path

| # | Venue | Why | Fit Score |
|---|---|---|---|
| 9 | **AsiaCCS** | Lower bar than Big 4 security. AsmDepictor published here. | 6/10 |
| 10 | **IEEE SANER** | Software analysis + reverse engineering. Niche but relevant. | 6/10 |
| 11 | **USENIX Security Workshops (e.g., WOOT)** | Workshop paper as stepping stone. | 5/10 |

### My Recommendation

**Primary target: NDSS or USENIX Security** -- Both have published multiple function naming papers. Our contributions (multi-context gated fusion, ext-call paradox, 300K BAP dataset, k-NN hybrid) align well with their scope. NDSS has the most recent activity in this space (2 papers in 2025).

**Backup: ISSTA or FSE** -- If the security angle is weaker, these SE venues value ablation methodology and would appreciate the tokenization + fusion architecture contributions.

**Journal option: ACM TOSEM or IEEE TSE** -- Extended version with full ablation + dataset release. Good for comprehensive evaluation papers.
