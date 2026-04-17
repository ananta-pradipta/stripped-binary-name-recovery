# Pipeline: Input/Output Examples for Every Step

This package contains realistic example data showing the exact input and output format for each step in the function name recovery pipeline.

All examples follow one primary function: `socket_init` at address `0x401230`, traced through the entire pipeline from C source code to final evaluation metrics.

---

## Directory Structure

```
step01_source/
  └── net_utils.c                          # Source code + compilation commands

step02_stripped/
  └── labels_net_utils.json                # Ground truth: address → function name

step03_bap_ir/
  └── output.bir                           # BAP intermediate representation

step04_cfg_graphs/
  ├── net_utils_0x401230.json              # CFG + normalized tokens (socket_init)
  └── net_utils_0x401400.json              # CFG for compute_checksum (no ext calls)

step05_external_calls/
  ├── external_calls_net_utils.json        # External call lists per function
  └── external_vocab.json                  # External function vocabulary (nn.Embedding)

step06_bpe_vocab/
  └── bpe_vocabulary.json                  # BPE vocabulary + encoding examples

step07_block_encoder/
  └── block_embeddings_0x401230.json       # Per-block embeddings (5 blocks → 5 vectors)

step08_graph_encoder/
  └── graph_encoder_output_0x401230.json   # GAT + attention pooling → f ∈ ℝ⁵¹²

step09_external_encoder/
  └── external_encoder_output_0x401230.json # Bi-GRU over ext calls → c ∈ ℝ⁵¹²

step10_gated_fusion/
  └── gated_fusion_output_0x401230.json    # Gate + fusion → z ∈ ℝ⁵¹² (with gate analysis)

step11_decoder/
  └── decoder_output_0x401230.json         # Teacher forcing (train) + beam search (inference)

step12_evaluation/
  └── evaluation_results.json              # Per-function metrics + ablation table + error analysis
```

## Pipeline Flow

```
Source Code  →  Compile (-g -O2)  →  Strip  →  BAP Lift  →  Parse CFG + Tokens
     [Step 1]       [Step 2]         [Step 2]    [Step 3]      [Step 4 + 5]
                                                                    ↓
Block Encoder  →  Graph Encoder  →  Gated Fusion (Option B)  →  GRU Decoder  →  Evaluation
   [Step 7]         [Step 8]           [Step 9+10]              [Step 11]       [Step 12]
```

## Key Function Examples

| Function | Address | Ext Calls | Gate (g) | Predicted | True | F1 |
|----------|---------|-----------|----------|-----------|------|----|
| socket_init | 0x401230 | socket,bind,listen | 0.35 | socket_init | socket_init | 1.0 |
| parse_config | 0x4012a0 | fopen,fgets,fclose | 0.42 | read_config | parse_config | 0.67 |
| handle_client | 0x401350 | read,write,close | 0.38 | handle_client | handle_client | 1.0 |
| compute_checksum | 0x401400 | (none) | 0.92 | calc_hash | compute_checksum | 0.0 |

Note: compute_checksum has no external calls, so the gate pushes g→0.92 (trust code).
The prediction is semantically close but has zero subtoken overlap.
