# Cross-Project Evaluation Dataset

Stripped x86-64 ELF binaries for cross-project evaluation. These packages are **completely absent from training data**.

## Packages

| Package | Description | Binaries | Opt Levels | GT Available |
|---------|-------------|----------|------------|--------------|
| tengine | Nginx fork (Alibaba) | 1 (nginx) | O0, O2 | Yes |
| angie | Nginx fork (Russian) | 1 (angie) | O0, O1, O2, O3 | Yes |
| recutils | GNU record utilities | 9 tools | O0, O1, O2, O3 | Yes |
| nginx118 | Nginx 1.18 LTS | 1 (nginx118) | O0, O1, O2, O3 | Yes |

## Directory Structure

```
stripped/          # Stripped binaries (no symbols)
  tengine_nginx_O0
  tengine_nginx_O2
  angie_angie_O0_stripped
  angie_angie_O1_stripped
  ...
  recutils_csv2rec_O0_stripped
  recutils_recsel_O3_stripped
  ...

ground_truth/      # Function address -> name mappings (from debug binaries via nm)
  ground_truth.json
```

## Ground Truth Format

`ground_truth.json` maps binary name -> metadata:

```json
{
  "tengine_nginx_O0": {
    "package": "tengine",
    "binary": "nginx_O0",
    "num_functions": 1753,
    "functions": {
      "0000000000403ac0": "ngx_cpuinfo",
      "0000000000403b30": "ngx_os_init",
      ...
    }
  }
}
```

Total: 46 binaries with ground truth, 18,595 functions (tengine: 3,184, recutils: 5,109, nginx118: 4,865, angie: 5,437).

## Notes

- **angie / nginx118**: Debug binaries recovered from the SymLM `dataset_generation/xproj_debug` directory; matched to existing stripped binaries by BuildID. Ground truth extracted via `nm --defined-only` (T/t/W/w symbols).
- Addresses are hex strings from `nm` (T/t symbols = text/code section)
- For StarCoder comparison: disassemble stripped binaries, predict function names, compare against ground_truth.json
