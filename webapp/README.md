# Function Name Recovery Web App

A Streamlit web application for predicting human-readable function names from
stripped ELF binaries using graph neural networks.

**Anonymous submission — CCS 2026**


## Features

- **Package Analysis**: Upload a source tarball or URL, compile with debug info,
  strip the binaries, predict function names, and compare against ground truth.
- **Stripped Binary Prediction**: Upload one or more pre-stripped binaries and get
  predicted function names from all models side by side.
- **Multi-model Comparison**: All 4 DL model variants run in parallel with
  aggregate metrics and per-function comparison tables.
- **Pre-computed Demo**: Ablation study results and diffutils demo always available.
- **Training Data Info**: Sidebar shows all 26 packages / 66 binaries used in training.


## Project Structure

The webapp lives inside the project root and uses **symlinks** to avoid
duplicating code, data, and model weights:

```
bfnr-project/
  src/                 <-- shared source code
  data/                <-- shared data (BPE model, ext vocab, match index)
  checkpoints/         <-- trained model weights
  configs/             <-- YAML configs
  demo/                <-- pre-computed demo results
  results/             <-- ablation results
  webapp/              <-- this directory
    app.py             # Streamlit application
    inference.py       # Multi-model inference engine
    rebuild_ext_vocab.py  # Fix external vocab mismatch
    download_checkpoints.py
    setup.sh           # Creates symlinks to parent dirs
    run.sh             # One-command startup
    requirements.txt
    Dockerfile
    Dockerfile.lite
    .streamlit/config.toml
```

After running `setup.sh`, the webapp directory contains symlinks:

```
webapp/
  src -> ../src
  data -> ../data
  checkpoints -> ../checkpoints
  configs -> ../configs
  demo -> ../demo
  results -> ../results
```

This means there is exactly **one copy** of everything. Changes to source code,
data, or checkpoints in the project root are immediately visible to the webapp.


## Quick Start

```bash
cd bfnr-project/webapp
bash setup.sh         # creates symlinks (one time)
pip install -r requirements.txt
bash run.sh           # or: streamlit run app.py
```

Open http://localhost:8501 in your browser.


## External Vocab Mismatch

If the sidebar shows a warning about external vocab mismatch (e.g.,
"model=605, on-disk=120"), the models that use external calls (Option A,
Option B) will produce degraded results. To fix:

```bash
cd bfnr-project
python webapp/rebuild_ext_vocab.py
```

This rebuilds `data/external_calls/external_vocab.json` from the per-binary
`*_external.json` files generated during training. The webapp picks it up
automatically via the symlink.


## Deployment

### Docker (Full, with BAP)

For deployment, copy (not symlink) the shared directories into the webapp:

```bash
cd bfnr-project
mkdir -p webapp_deploy
cp webapp/*.py webapp_deploy/
cp webapp/*.sh webapp_deploy/
cp webapp/*.txt webapp_deploy/
cp webapp/Dockerfile webapp_deploy/
cp -r webapp/.streamlit webapp_deploy/
cp -r src webapp_deploy/
cp -r data webapp_deploy/
cp -r checkpoints webapp_deploy/
cp -r configs webapp_deploy/
cp -r demo webapp_deploy/
cp -r results webapp_deploy/

cd webapp_deploy
docker build -t funcname-recovery .
docker run -p 7860:7860 funcname-recovery
```

### Hugging Face Spaces

Same idea: create a standalone copy with all files, push to the Space repo
with model weights tracked via Git LFS.


## Troubleshooting

**"No models loaded"**: Check that the `checkpoints` symlink points to the
right directory and contains `.pt` files.

**"BAP not detected"**: Install BAP via opam. The app still works in demo
mode without BAP.

**Ext vocab warning**: Run `rebuild_ext_vocab.py` from the project root.

**Slow first load**: Models are cached after first load. Subsequent page
refreshes are fast.
