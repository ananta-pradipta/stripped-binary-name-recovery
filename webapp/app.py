"""
Function Name Recovery in Stripped Binaries
Web Demo -- CS 785 Advanced Binary Analysis w/ Machine Learning
"""
import json
import os
import sys
import tempfile
import subprocess

import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference import (
    ModelManager, InferenceEngine,
    extract_ground_truth_from_debug_binary,
    load_trained_packages_info,
)
from src.evaluation.metrics import compute_all_metrics, normalize_name

# --------------------------------------------------------------------------- #
#  Page config
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="Function Name Recovery",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    :root { --njit-red: #D32F2F; --njit-dark: #1a1a2e; }
    .main-header {
        background: linear-gradient(135deg, #D32F2F 0%, #8B0000 100%);
        color: white; padding: 1.5rem 2rem; border-radius: 10px; margin-bottom: 1.5rem;
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.8rem; }
    .main-header p { color: rgba(255,255,255,0.85); margin: 0.3rem 0 0 0; }
    div[data-testid="stExpander"] { border: 1px solid #e0e0e0; border-radius: 8px; }

    /* Compact dataframe tables */
    div[data-testid="stDataFrame"] table {
        font-size: 0.78rem !important;
    }
    div[data-testid="stDataFrame"] th {
        font-size: 0.78rem !important;
        padding: 4px 8px !important;
        white-space: nowrap;
    }
    div[data-testid="stDataFrame"] td {
        font-size: 0.78rem !important;
        padding: 3px 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
#  Cached loading
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner=False)
def load_models():
    manager = ModelManager()
    loaded = manager.load_all()
    engine = InferenceEngine(manager)
    return manager, engine, loaded

@st.cache_data
def load_ablation_results():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "ablation_table.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

@st.cache_data
def load_demo_comparison():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo", "results", "comparison_diff.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

@st.cache_data
def get_trained_packages():
    return load_trained_packages_info()

# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def check_bap_installed() -> bool:
    try:
        result = subprocess.run(["bap", "--version"], capture_output=True, timeout=10)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

def build_comparison_df(evaluation: dict) -> pd.DataFrame:
    # Show only GAT + Option B results in the per-function table
    optb_key = None
    for model_name in evaluation:
        if "Option B" in model_name:
            optb_key = model_name
            break

    if not optb_key or not evaluation[optb_key]["per_function"]:
        return pd.DataFrame()

    rows = []
    for func in evaluation[optb_key]["per_function"]:
        rows.append({
            "Address": func["address"],
            "True Name": func["true_name"],
            "Predicted Name": func["predicted"],
            "F1": round(func["f1"], 3),
        })

    return pd.DataFrame(rows)

def build_prediction_df(predictions: dict) -> pd.DataFrame:
    all_funcs = {}
    for model_name, preds in predictions.items():
        for pred in preds:
            addr = pred["address"]
            if addr not in all_funcs:
                all_funcs[addr] = {
                    "Address": addr,
                    "BAP Name": pred["bap_name"],
                    "Blocks": pred["num_blocks"],
                    "Ext Calls": pred["num_ext_calls"],
                }
            all_funcs[addr][model_name] = pred["predicted_name"]
    rows = list(all_funcs.values())
    rows.sort(key=lambda r: int(r["Address"], 16) if r["Address"].startswith("0x") else 0)
    return pd.DataFrame(rows)


OPTION_B_COL_STYLE = {"background-color": "rgba(211, 47, 47, 0.08)", "font-weight": "600"}
OPTION_B_ROW_CSS = "background-color: rgba(211, 47, 47, 0.08); font-weight: 600"


def style_highlight_option_b_column(df: pd.DataFrame):
    """Highlight GAT + Option B column in prediction/comparison tables."""
    optb_cols = [c for c in df.columns if "Option B" in c]
    if optb_cols:
        return df.style.set_properties(subset=optb_cols, **OPTION_B_COL_STYLE)
    return df.style


def style_highlight_option_b_row(df: pd.DataFrame):
    """Highlight the GAT + Option B row in metrics/ablation tables."""
    def _apply(row):
        if "Option B" in str(row.get("Model", "")):
            return [OPTION_B_ROW_CSS] * len(row)
        return [""] * len(row)
    return df.style.apply(_apply, axis=1)

# --------------------------------------------------------------------------- #
#  Sidebar
# --------------------------------------------------------------------------- #

def render_sidebar(manager, loaded_models):
    with st.sidebar:
        st.markdown("### Settings")
        beam_width = st.slider(
            "Beam Width", min_value=1, max_value=10, value=5,
            help="Beam search width for name generation.",
        )

        bap_available = check_bap_installed()
        if bap_available:
            st.success("BAP is installed")
        else:
            st.warning("BAP not detected. Live binary analysis requires BAP.")

        if manager.vocab_warnings:
            st.markdown("---")
            st.markdown("### Warnings")
            for w in manager.vocab_warnings:
                st.warning(w)

        st.markdown("---")
        st.markdown("### Loaded Models")
        if not loaded_models:
            st.error("No models loaded. Run `python download_checkpoints.py` first.")
        else:
            for info in manager.get_model_info():
                is_primary = "Option B" in info["name"]
                badge = "(primary)" if is_primary else ""
                with st.expander(f"{info['name']} {badge}", expanded=False):
                    st.caption(info["description"])
                    st.text(f"Parameters: {info['params']}")
                    st.text(f"Block Encoder: {info['block_encoder']}")
                    st.text(f"Graph Encoder: {info['graph_encoder']}")
                    st.text(f"Fusion: {info['fusion']}")
                    if info.get("vocab_mismatch"):
                        st.warning(f"Ext vocab mismatch: {info['vocab_info']}")

        st.markdown("---")
        st.markdown("### Training Data")
        pkg_info = get_trained_packages()
        if pkg_info["total_functions"] > 0:
            st.caption(
                f"{pkg_info['total_functions']:,} functions from "
                f"{pkg_info['total_binaries']} binaries across "
                f"{len(pkg_info['packages'])} packages"
            )
            with st.expander("View trained packages", expanded=False):
                for pkg_name in sorted(pkg_info["packages"]):
                    pkg = pkg_info["packages"][pkg_name]
                    bin_list = ", ".join(sorted(pkg["binaries"].keys()))
                    st.markdown(
                        f"**{pkg_name}** ({pkg['total_functions']} funcs): {bin_list}"
                    )
        else:
            st.caption("No training data info available.")

        st.markdown("---")
        st.markdown("### About")
        st.caption(
            "CS 785 Advanced Binary Analysis w/ Machine Learning\n\n"
            "Contributors: Ananta Pradipta, Robert Blacha, Zhihao Lin\n\n"
            "Reproducing and extending the DeBin with "
            "GNN-based deep learning for binary function name recovery."
        )

    return beam_width

# --------------------------------------------------------------------------- #
#  Tab 1: Package Analysis
# --------------------------------------------------------------------------- #

def render_package_tab(engine, loaded_models, beam_width):
    st.markdown("### Package Analysis")
    st.markdown(
        "Upload a source tarball or provide a URL. The system will compile the package, "
        "extract ground truth function names from the debug binary, strip the binary, "
        "then predict function names with all models and compare against ground truth."
    )

    bap_ok = check_bap_installed()
    if not bap_ok or not loaded_models:
        st.info(
            "Live analysis requires BAP and loaded model checkpoints. "
            "See the Demo Results tab for pre-computed results."
        )
        return

    col1, col2 = st.columns(2)
    with col1:
        uploaded_tarball = st.file_uploader(
            "Upload source tarball (.tar.gz, .tar.bz2, .tar.xz)",
            type=["gz", "bz2", "xz", "tar"], key="tarball_upload",
        )
    with col2:
        package_url = st.text_input(
            "Or enter package URL",
            placeholder="https://ftp.gnu.org/gnu/coreutils/coreutils-9.4.tar.xz",
            key="package_url",
        )

    if uploaded_tarball or package_url:
        if st.button("Analyze Package", type="primary"):
            with st.spinner("Running analysis pipeline..."):
                try:
                    results = run_package_analysis(engine, uploaded_tarball, package_url, beam_width)
                    display_evaluation_results(results)
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

def run_package_analysis(engine, uploaded_file, url, beam_width):
    work_dir = tempfile.mkdtemp()
    progress = st.progress(0, text="Starting...")
    try:
        progress.progress(10, text="Preparing source...")
        if uploaded_file:
            src_path = os.path.join(work_dir, uploaded_file.name)
            with open(src_path, "wb") as f:
                f.write(uploaded_file.read())
        elif url:
            src_path = os.path.join(work_dir, "package.tar.gz")
            subprocess.run(["wget", "-q", "-O", src_path, url], timeout=120, check=True)
        else:
            raise ValueError("No input provided")

        progress.progress(20, text="Extracting...")
        subprocess.run(["tar", "xf", src_path, "-C", work_dir], timeout=60, check=True)
        dirs = [d for d in os.listdir(work_dir) if os.path.isdir(os.path.join(work_dir, d))]
        if not dirs:
            raise RuntimeError("No directory found after extraction")
        src_dir = os.path.join(work_dir, dirs[0])

        progress.progress(30, text="Compiling with debug symbols...")
        env = os.environ.copy()
        env["CFLAGS"] = "-g -O2"
        env["CXXFLAGS"] = "-g -O2"
        configure_path = os.path.join(src_dir, "configure")
        if os.path.exists(configure_path):
            subprocess.run(["./configure", f"--prefix={work_dir}/install"],
                           cwd=src_dir, env=env, capture_output=True, timeout=120)
            subprocess.run(["make", "-j4"], cwd=src_dir, env=env, capture_output=True, timeout=300)

        progress.progress(50, text="Finding binaries...")
        binaries = []
        for root, _, files in os.walk(src_dir):
            for f in files:
                fpath = os.path.join(root, f)
                try:
                    result = subprocess.run(["file", fpath], capture_output=True, text=True, timeout=5)
                    if "ELF" in result.stdout and "executable" in result.stdout:
                        binaries.append(fpath)
                except (subprocess.TimeoutExpired, OSError):
                    pass
        if not binaries:
            raise RuntimeError("No ELF binaries found after compilation")

        all_results = {}
        for i, bin_path in enumerate(binaries):
            bin_name = os.path.basename(bin_path)
            progress.progress(50 + int(40 * i / len(binaries)), text=f"Analyzing {bin_name}...")
            gt = extract_ground_truth_from_debug_binary(bin_path)
            stripped_path = bin_path + ".stripped"
            subprocess.run(["strip", "-o", stripped_path, bin_path], check=True)
            result = engine.predict_all_models(stripped_path, beam_width)
            evaluation = engine.evaluate_with_ground_truth(result["predictions"], gt)
            all_results[bin_name] = {
                "stats": result["stats"], "evaluation": evaluation,
                "ground_truth_count": len(gt),
            }
        progress.progress(100, text="Done!")
        return all_results
    finally:
        import shutil
        shutil.rmtree(work_dir, ignore_errors=True)

def display_evaluation_results(all_results):
    for bin_name, data in all_results.items():
        st.markdown(f"#### Binary: `{bin_name}`")
        st.caption(
            f"Total functions: {data['stats']['total_functions']} | "
            f"Targets (sub_XXXX, 2+ blocks): {data['stats']['target_functions']} | "
            f"Ground truth available: {data['ground_truth_count']}"
        )
        if data["evaluation"]:
            metrics_data = []
            for model_name, ev in data["evaluation"].items():
                metrics_data.append({
                    "Model": model_name,
                    "F1": round(ev["f1"], 4),
                    "Precision": round(ev["precision"], 4),
                    "Recall": round(ev["recall"], 4),
                    "Exact Match": f"{ev['exact_match']:.1%}",
                    "Edit Sim": round(ev["edit_sim"], 4),
                    "Evaluated": ev["num_evaluated"],
                })
            st.dataframe(style_highlight_option_b_row(pd.DataFrame(metrics_data)), use_container_width=True, hide_index=True)
            comparison_df = build_comparison_df(data["evaluation"])
            if not comparison_df.empty:
                with st.expander("Per-function comparison (GAT + Option B)", expanded=False):
                    st.dataframe(comparison_df, use_container_width=True, height=400, hide_index=True)

# --------------------------------------------------------------------------- #
#  Tab 2: Stripped Binary Prediction (multi-file)
# --------------------------------------------------------------------------- #

def render_binary_tab(engine, loaded_models, beam_width):
    st.markdown("### Stripped Binary Prediction")
    st.markdown(
        "Upload one or more stripped ELF binaries. The system will extract the control "
        "flow graph using BAP, then predict function names with all loaded models."
    )

    bap_ok = check_bap_installed()
    if not bap_ok:
        st.warning("BAP is required for binary analysis but is not installed.")
        st.code("opam install bap", language="bash")
        return
    if not loaded_models:
        st.error("No model checkpoints loaded. Run `python download_checkpoints.py` first.")
        return

    uploaded_binaries = st.file_uploader(
        "Upload stripped ELF binary (or multiple)",
        type=None, key="binary_upload", accept_multiple_files=True,
        help="Upload one or more stripped x86-64 ELF binaries",
    )

    with st.expander("Optional: upload debug binaries for evaluation"):
        st.caption(
            "Upload the corresponding debug (unstripped) versions to evaluate "
            "predictions against ground truth. Files are matched by filename prefix."
        )
        debug_binaries = st.file_uploader(
            "Upload debug (unstripped) binaries",
            type=None, key="debug_upload", accept_multiple_files=True,
        )

    if uploaded_binaries:
        if st.button("Predict Function Names", type="primary"):
            _run_multi_binary_prediction(engine, uploaded_binaries, debug_binaries or [], beam_width)

def _match_debug_to_stripped(stripped_name: str, debug_files: list) -> object:
    base = stripped_name.lower()
    for suffix in ("_stripped", ".stripped", "-stripped", "_strip"):
        base = base.replace(suffix, "")
    base = os.path.splitext(base)[0]

    for dbg in debug_files:
        dbg_base = dbg.name.lower()
        for suffix in ("_debug", ".debug", "-debug", "_dbg", "_unstripped"):
            dbg_base = dbg_base.replace(suffix, "")
        dbg_base = os.path.splitext(dbg_base)[0]
        if base == dbg_base or base.startswith(dbg_base) or dbg_base.startswith(base):
            return dbg
    return None

def _run_multi_binary_prediction(engine, uploaded_binaries, debug_binaries, beam_width):
    total = len(uploaded_binaries)
    for idx, uploaded_binary in enumerate(uploaded_binaries):
        st.markdown("---")
        st.markdown(f"#### Binary {idx + 1}/{total}: `{uploaded_binary.name}`")
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_binary.name}") as tmp:
                tmp.write(uploaded_binary.read())
                tmp_path = tmp.name

            file_result = subprocess.run(["file", tmp_path], capture_output=True, text=True)
            if "ELF" not in file_result.stdout:
                st.error(f"Not an ELF binary. Detected: {file_result.stdout.strip()}")
                os.unlink(tmp_path)
                continue

            status = st.empty()
            result = engine.predict_all_models(tmp_path, beam_width, progress_callback=lambda msg: status.text(msg))
            status.empty()

            stats = result["stats"]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Functions", stats["total_functions"])
            with col2:
                st.metric("Target Functions", stats["target_functions"])
            with col3:
                st.metric("Models Used", len(stats["models_used"]))

            pred_df = build_prediction_df(result["predictions"])
            if not pred_df.empty:
                st.dataframe(style_highlight_option_b_column(pred_df), use_container_width=True, height=400, hide_index=True)
                st.download_button(
                    f"Download predictions for {uploaded_binary.name} (JSON)",
                    json.dumps({uploaded_binary.name: result["predictions"]}, indent=2),
                    file_name=f"predictions_{uploaded_binary.name}.json",
                    mime="application/json", key=f"dl_{idx}",
                )
            else:
                st.warning("No target functions found (sub_XXXX with 2+ blocks).")

            matched_debug = _match_debug_to_stripped(uploaded_binary.name, debug_binaries)
            if matched_debug:
                st.markdown(f"**Evaluating against debug binary:** `{matched_debug.name}`")
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{matched_debug.name}") as dbg_tmp:
                    dbg_tmp.write(matched_debug.read())
                    dbg_path = dbg_tmp.name
                gt = extract_ground_truth_from_debug_binary(dbg_path)
                if gt:
                    evaluation = engine.evaluate_with_ground_truth(result["predictions"], gt)
                    st.markdown("**Evaluation Against Ground Truth**")
                    display_evaluation_results({
                        uploaded_binary.name: {
                            "stats": stats, "evaluation": evaluation,
                            "ground_truth_count": len(gt),
                        }
                    })
                else:
                    st.warning("Could not extract ground truth. Ensure binary was compiled with -g.")
                os.unlink(dbg_path)
                matched_debug.seek(0)

            os.unlink(tmp_path)
        except RuntimeError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Prediction failed for {uploaded_binary.name}: {e}")
            import traceback
            st.code(traceback.format_exc())

# --------------------------------------------------------------------------- #
#  Tab 3: Architecture
# --------------------------------------------------------------------------- #

def render_architecture_tab():
    st.markdown("### Pipeline Architecture")
    st.markdown("""
The inference pipeline transforms a stripped binary into predicted function names
through several stages:

**Stage 1: Binary Lifting.**
The stripped binary is lifted to BAP-IR (Binary Analysis Platform Intermediate
Representation), which provides a normalized view of the binary's instructions.

**Stage 2: CFG Extraction + Tokenization.**
BAP-IR is parsed into per-function Control Flow Graphs. Each basic block's
instructions are classified into semantic instruction types (approximately 20
categories like `CALL_malloc`, `MEM_WRITE`, `COND_BRANCH`, `STACK_OP`, etc.)
rather than using raw BAP-IR tokens. This reduces the vocabulary from 32,000+
to around 120 meaningful types.

**Stage 3: Block Encoding.**
Each basic block's token sequence is encoded into a fixed-size vector. Two
variants are used: a Transformer encoder (for GAT models) and mean pooling
(for GCN baseline).

**Stage 4: Graph Encoding.**
The CFG is processed by a Graph Neural Network. GCN uses simple message passing
with mean pooling, while GAT uses multi-head attention with learned attention pooling.

**Stage 5: External Call Encoding + Fusion.**
External library calls (which survive stripping via PLT/GOT entries) are encoded
using a bidirectional GRU. The external call context is fused with the graph
embedding using one of several strategies: no fusion (GCN + Mean Pool,
GAT + Attn Pool), Option A (node-level injection before GNN), or
Option B (gated fusion: z = g * f + (1 - g) * c).

**Stage 6: Decoding.**
A GRU decoder with beam search generates BPE sub-tokens, which are decoded into
the final predicted function name.
""")

    st.markdown("#### Model Variants")
    variants_data = [
        {"Variant": "GCN + Mean Pool", "Block Encoder": "Mean Pooling", "Graph Encoder": "GCN",
         "Pooling": "Mean", "External Calls": "No", "Fusion": "None"},
        {"Variant": "GAT + Attn Pool", "Block Encoder": "Transformer", "Graph Encoder": "GAT",
         "Pooling": "Attention", "External Calls": "No", "Fusion": "None"},
        {"Variant": "GAT + Option A", "Block Encoder": "Transformer", "Graph Encoder": "GAT",
         "Pooling": "Attention", "External Calls": "Yes", "Fusion": "Node Features"},
        {"Variant": "GAT + Option B (primary)", "Block Encoder": "Transformer", "Graph Encoder": "GAT",
         "Pooling": "Attention", "External Calls": "Yes", "Fusion": "Gated"},
        {"Variant": "ExtraTrees (baseline)", "Block Encoder": "Feature Eng.", "Graph Encoder": "N/A",
         "Pooling": "N/A", "External Calls": "Indirect", "Fusion": "N/A"},
    ]
    st.dataframe(pd.DataFrame(variants_data), use_container_width=True, hide_index=True)

    st.markdown("#### Evaluation Metrics")
    st.markdown("""
- **Sub-token F1**: Function names are split into sub-tokens (e.g., `close_stdout`
  becomes `[close, stdout]`). Precision, recall, and F1 are computed over the sub-token bags.
- **Exact Match**: Whether the normalized predicted name matches the ground truth.
- **Character N-gram Similarity**: Dice coefficient over character trigrams.
- **Edit Similarity**: Normalized Levenshtein distance (1.0 = identical, 0.0 = completely different).
""")

# --------------------------------------------------------------------------- #
#  Tab 4: Demo Results
# --------------------------------------------------------------------------- #

def render_demo_tab():
    st.markdown("### Pre-computed Results")
    demo = load_demo_comparison()
    ablation = load_ablation_results()

    if not demo and not ablation:
        st.info("No pre-computed results found.")
        return

    if ablation:
        st.markdown("#### Ablation Study Results (Test Set)")
        rows = []
        for model_name, metrics in ablation.items():
            display_name = model_name.replace("\u2605 ", "")
            rows.append({
                "Model": display_name,
                "Precision": round(metrics.get("precision", 0), 4),
                "Recall": round(metrics.get("recall", 0), 4),
                "F1": round(metrics.get("f1", 0), 4),
                "Exact Match": f"{metrics.get('exact_match', 0):.1%}",
                "Char N-gram Sim": round(metrics.get("char_ngram_sim", 0), 4),
                "Edit Sim": round(metrics.get("edit_sim", 0), 4),
            })
        st.dataframe(style_highlight_option_b_row(pd.DataFrame(rows)), use_container_width=True, hide_index=True)
        st.caption(
            "Note: ExtraTrees is a closed-vocab classification baseline. "
            "Its metrics are not directly comparable to the generative DL models' sub-token F1."
        )

    if demo:
        st.markdown("---")
        st.markdown(f"#### Demo on Unseen Package: `{demo.get('package', 'diffutils')}`")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Predicted", demo.get("total_predicted", "N/A"))
        with col2:
            st.metric("With Ground Truth", demo.get("total_with_gt", "N/A"))
        with col3:
            st.metric("Exact Matches", demo.get("exact_matches", "N/A"))
        with col4:
            st.metric("Avg F1", f"{demo.get('avg_f1', 0):.3f}")

        funcs = demo.get("functions", [])
        if funcs:
            func_df = pd.DataFrame(funcs).rename(columns={
                "address": "Address", "predicted": "Predicted", "true_name": "True Name",
                "f1": "F1", "em": "Exact Match", "ng": "N-gram Sim",
                "ed": "Edit Sim", "ext": "Ext Calls",
            })
            display_cols = [c for c in ["Address", "True Name", "Predicted", "F1", "Exact Match", "Edit Sim"]
                           if c in func_df.columns]
            st.dataframe(func_df[display_cols], use_container_width=True, height=500, hide_index=True)

# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    st.markdown("""
    <div class="main-header">
        <h1>Function Name Recovery in Stripped Binaries</h1>
        <p>Using Graph Neural Networks and Deep Learning with Gated External Call Fusion</p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading models..."):
        manager, engine, loaded_models = load_models()

    beam_width = render_sidebar(manager, loaded_models)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Stripped Binary", "Package Analysis", "Demo Results", "Architecture",
    ])

    with tab1:
        render_binary_tab(engine, loaded_models, beam_width)
    with tab2:
        render_package_tab(engine, loaded_models, beam_width)
    with tab3:
        render_demo_tab()
    with tab4:
        render_architecture_tab()

if __name__ == "__main__":
    main()
