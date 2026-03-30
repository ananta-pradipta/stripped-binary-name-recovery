"""
Multi-model inference engine for the web app.

Loads all available model variants and runs prediction in parallel.

NOTE: This file expects src/, data/, checkpoints/, etc. to be accessible
from the webapp directory (either via symlinks or direct copies).
Run setup.sh to create symlinks to the parent project directory.
"""
import json
import os
import re
import tempfile
import subprocess
from typing import Dict, List, Optional, Tuple

import torch
import sentencepiece as spm

from src.models.function_namer import FunctionNamer
from src.preprocessing.parse_bap import parse_bir_file
from src.evaluation.metrics import compute_all_metrics, normalize_name

# --------------------------------------------------------------------------- #
#  Model registry
# --------------------------------------------------------------------------- #

MODEL_REGISTRY = {
    "GCN + Mean Pool": {
        "checkpoint": "checkpoints/variant2_gcn/best_model.pt",
        "description": "GCN graph encoder with mean pooling (no external calls)",
    },
    "GAT + Attn Pool": {
        "checkpoint": "checkpoints/variant3_gat/best_model.pt",
        "description": "GAT graph encoder with attention pooling (no external calls)",
    },
    "GAT + Option A": {
        "checkpoint": "checkpoints/variant4_optA/best_model.pt",
        "description": "GAT with node-level external call injection",
    },
    "GAT + Option B": {
        "checkpoint": "checkpoints/best_model.pt",
        "description": "GAT with gated fusion of code + external calls (primary)",
    },
}

EXTRATREES_CHECKPOINT = "checkpoints/extratrees_model.pkl"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BPE_MODEL_PATH = os.path.join(BASE_DIR, "data", "bpe_model", "bpe.model")
EXT_VOCAB_PATH = os.path.join(BASE_DIR, "data", "external_calls", "external_vocab.json")


def load_trained_packages_info() -> dict:
    """Load info about packages used in training from match_index.json."""
    match_index_path = os.path.join(BASE_DIR, "data", "match_index.json")
    if not os.path.exists(match_index_path):
        return {"packages": {}, "total_functions": 0, "total_binaries": 0}

    with open(match_index_path) as f:
        mi = json.load(f)

    packages = {}
    for gf, info in mi.items():
        binary = info.get("binary", "unknown")
        parts = binary.split("_", 1)
        pkg = parts[0] if parts else binary
        bin_name = parts[1] if len(parts) > 1 else binary

        if pkg not in packages:
            packages[pkg] = {"binaries": {}, "total_functions": 0}
        if bin_name not in packages[pkg]["binaries"]:
            packages[pkg]["binaries"][bin_name] = 0
        packages[pkg]["binaries"][bin_name] += 1
        packages[pkg]["total_functions"] += 1

    return {
        "packages": packages,
        "total_functions": len(mi),
        "total_binaries": sum(len(p["binaries"]) for p in packages.values()),
    }


class ModelManager:
    """Manages loading and inference for all model variants."""

    def __init__(self):
        self.models: Dict[str, dict] = {}
        self.sp = None
        self.ext_vocab = None
        self.ext_vocab_size_on_disk = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded = False
        self.vocab_warnings: List[str] = []

    def discover_checkpoints(self) -> Dict[str, str]:
        available = {}
        for name, info in MODEL_REGISTRY.items():
            ckpt_path = os.path.join(BASE_DIR, info["checkpoint"])
            if os.path.exists(ckpt_path):
                available[name] = ckpt_path
        return available

    def load_all(self, progress_callback=None) -> List[str]:
        if self._loaded:
            return list(self.models.keys())

        loaded_names = []

        if progress_callback:
            progress_callback("Loading tokenizer...")
        votes_path = os.path.join(BASE_DIR, "data", "votes_vocab.json")
        if os.path.exists(votes_path):
            from src.preprocessing.build_votes import VotesTokenizer
            self.sp = VotesTokenizer(vocab_path=votes_path)
        else:
            self.sp = spm.SentencePieceProcessor(model_file=BPE_MODEL_PATH)

        with open(EXT_VOCAB_PATH) as f:
            ext_data = json.load(f)
        self.ext_vocab = ext_data["vocabulary"]
        self.ext_vocab_size_on_disk = ext_data["vocab_size"]

        available = self.discover_checkpoints()
        total = len(available)
        for i, (name, ckpt_path) in enumerate(available.items()):
            if progress_callback:
                progress_callback(f"Loading {name} ({i + 1}/{total})...")
            try:
                model_info = self._load_single_model(name, ckpt_path)
                model_info["description"] = MODEL_REGISTRY[name]["description"]
                self.models[name] = model_info
                loaded_names.append(name)
            except Exception as e:
                print(f"  Failed to load {name}: {e}")

        et_path = os.path.join(BASE_DIR, EXTRATREES_CHECKPOINT)
        if os.path.exists(et_path):
            if progress_callback:
                progress_callback("Loading ExtraTrees baseline...")
            try:
                import joblib
                et_model = joblib.load(et_path)
                self.models["ExtraTrees"] = {
                    "type": "extratrees",
                    "model": et_model,
                    "description": "DeBin ExtraTrees classification baseline",
                }
                loaded_names.append("ExtraTrees")
            except Exception as e:
                print(f"  Failed to load ExtraTrees: {e}")

        self._loaded = True
        return loaded_names

    def _load_single_model(self, name: str, ckpt_path: str) -> dict:
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        cfg = ckpt["config"]
        token_vocab = ckpt["token_vocab"]

        cfg["decoder"]["bpe_vocab_size"] = self.sp.get_piece_size()

        # Keep training-time ext vocab size from checkpoint config.
        training_ext_vocab_size = cfg.get("external_encoder", {}).get("vocab_size")
        if training_ext_vocab_size is None:
            cfg["external_encoder"]["vocab_size"] = self.ext_vocab_size_on_disk

        cfg["block_encoder"]["token_vocab_size"] = len(token_vocab)
        cfg["graph_encoder"]["input_dim"] = cfg["block_encoder"]["output_dim"]

        model = FunctionNamer(cfg).to(self.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        param_count = sum(p.numel() for p in model.parameters())

        # Use ext_vocab from checkpoint if available (guarantees correct ID mapping).
        # Fall back to on-disk vocab otherwise.
        uses_ext = cfg.get("external_encoder", {}).get("enabled", False)
        if "ext_vocab" in ckpt and uses_ext:
            ext_vocab = ckpt["ext_vocab"]
            ext_vocab_source = "checkpoint"
            print(f"  {name}: using ext_vocab from checkpoint ({len(ext_vocab)} tokens)")
        else:
            ext_vocab = self.ext_vocab
            ext_vocab_source = "disk"
            if uses_ext and training_ext_vocab_size and training_ext_vocab_size != self.ext_vocab_size_on_disk:
                warning = (
                    f"{name}: ext_vocab not in checkpoint, using on-disk vocab. "
                    f"ID mapping may differ from training "
                    f"(model={training_ext_vocab_size}, disk={self.ext_vocab_size_on_disk}). "
                    f"Run patch_checkpoints.py then restart."
                )
                self.vocab_warnings.append(warning)
                print(f"  WARNING: {warning}")

        return {
            "type": "dl",
            "model": model,
            "config": cfg,
            "token_vocab": token_vocab,
            "ext_vocab": ext_vocab,
            "ext_vocab_source": ext_vocab_source,
            "param_count": param_count,
            "training_ext_vocab_size": training_ext_vocab_size,
        }

    def get_model_names(self) -> List[str]:
        return list(self.models.keys())

    def get_model_info(self) -> List[dict]:
        info_list = []
        for name, m in self.models.items():
            entry = {"name": name, "description": m.get("description", "")}
            if m["type"] == "dl":
                entry["params"] = f'{m["param_count"]:,}'
                cfg = m["config"]
                entry["block_encoder"] = cfg["block_encoder"]["type"]
                entry["graph_encoder"] = cfg["graph_encoder"]["type"]
                entry["fusion"] = cfg["fusion"]["type"]
                entry["pooling"] = cfg["graph_encoder"].get("pooling", "mean")
                entry["uses_ext"] = cfg.get("external_encoder", {}).get("enabled", False)
                train_size = m.get("training_ext_vocab_size")
                vocab_source = m.get("ext_vocab_source", "disk")
                if vocab_source == "checkpoint":
                    entry["vocab_mismatch"] = False
                    entry["vocab_info"] = f"from checkpoint ({len(m.get('ext_vocab', {}))} tokens)"
                elif train_size and train_size != self.ext_vocab_size_on_disk and entry["uses_ext"]:
                    entry["vocab_mismatch"] = True
                    entry["vocab_info"] = f"trained={train_size}, on-disk={self.ext_vocab_size_on_disk}"
                else:
                    entry["vocab_mismatch"] = False
            else:
                entry["params"] = "N/A (sklearn)"
                entry["block_encoder"] = "feature engineering"
                entry["graph_encoder"] = "N/A"
                entry["fusion"] = "N/A"
                entry["pooling"] = "N/A"
                entry["uses_ext"] = False
                entry["vocab_mismatch"] = False
            info_list.append(entry)
        return info_list


class InferenceEngine:
    """Runs inference on a binary using the BAP pipeline + loaded models."""

    def __init__(self, manager: ModelManager):
        self.manager = manager

    def lift_binary(self, binary_path: str) -> Optional[str]:
        bir_path = tempfile.mktemp(suffix=".bir")
        try:
            result = subprocess.run(
                ["bap", binary_path, f"--dump=bir:{bir_path}"],
                capture_output=True,
                timeout=300,
            )
            if result.returncode != 0:
                stderr = result.stderr.decode("utf-8", errors="replace")
                raise RuntimeError(f"BAP failed (exit {result.returncode}): {stderr[:500]}")
            if not os.path.exists(bir_path) or os.path.getsize(bir_path) == 0:
                raise RuntimeError("BAP produced empty output")
            return bir_path
        except FileNotFoundError:
            raise RuntimeError(
                "BAP (Binary Analysis Platform) is not installed. "
                "Install with: opam install bap"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("BAP timed out after 300 seconds")

    def parse_functions(self, bir_path: str) -> Dict[str, dict]:
        return parse_bir_file(bir_path)

    def extract_external_calls(
        self, bir_path: str, binary_name: str = "_predict"
    ) -> Dict[str, List[str]]:
        ext_tmp_dir = tempfile.mkdtemp()
        try:
            subprocess.run(
                [
                    "python3", "-m", "src.preprocessing.extract_external",
                    "--bir", bir_path,
                    "--binary-name", binary_name,
                    "--output-dir", ext_tmp_dir,
                    "--vocab-path", os.path.join(ext_tmp_dir, "_throwaway_vocab.json"),
                ],
                capture_output=True,
                cwd=BASE_DIR,
            )
            ext_by_func = {}
            ext_file = os.path.join(ext_tmp_dir, f"{binary_name}_external.json")
            if os.path.exists(ext_file):
                with open(ext_file) as f:
                    ext_data = json.load(f)
                for func in ext_data.get("functions", []):
                    key = func.get("function_name", "")
                    calls = [c["name"] for c in func.get("external_calls", [])]
                    ext_by_func[key] = calls
            return ext_by_func
        finally:
            import shutil
            shutil.rmtree(ext_tmp_dir, ignore_errors=True)

    def predict_single_model(
        self,
        model_name: str,
        functions: Dict[str, dict],
        ext_by_func: Dict[str, List[str]],
        beam_width: int = 5,
    ) -> List[dict]:
        minfo = self.manager.models[model_name]
        if minfo["type"] != "dl":
            return []

        model = minfo["model"]
        cfg = minfo["config"]
        token_vocab = minfo["token_vocab"]
        sp = self.manager.sp
        ext_vocab = minfo.get("ext_vocab", self.manager.ext_vocab)
        device = self.manager.device

        sos_id = sp.bos_id()
        eos_id = sp.eos_id()
        max_blocks = cfg["data"]["max_blocks_per_function"]
        max_tokens = cfg["data"]["max_tokens_per_block"]

        targets = {
            name: data
            for name, data in functions.items()
            if name.startswith("sub_") and data["num_blocks"] >= 2
        }

        results = []
        for func_name, func_data in sorted(
            targets.items(),
            key=lambda x: (
                float("inf")
                if not x[1]["address"].startswith("0x")
                else int(x[1]["address"], 16)
            ),
        ):
            blocks = func_data["blocks"][:max_blocks]

            block_token_ids = []
            for block in blocks:
                ids = [
                    token_vocab.get(t, token_vocab.get("<UNK>", 1))
                    for t in block["tokens"][:max_tokens]
                ]
                ids += [0] * (max_tokens - len(ids))
                block_token_ids.append(ids)

            num_blocks = len(block_token_ids)
            while len(block_token_ids) < max_blocks:
                block_token_ids.append([0] * max_tokens)

            ext = ext_by_func.get(func_name, [])
            ext_ids = [ext_vocab.get(name, ext_vocab.get("<NO_EXT>", 0)) for name in ext]
            if not ext_ids:
                ext_ids = [0]

            bt = torch.tensor([block_token_ids], dtype=torch.long, device=device)
            ei_raw = func_data["edges"]
            ei_filtered = [[s, d] for s, d in ei_raw if s < num_blocks and d < num_blocks]
            if not ei_filtered:
                ei_filtered = [[0, 0]]
            ei = torch.tensor(ei_filtered, dtype=torch.long, device=device).t().contiguous()
            ec = torch.tensor([ext_ids], dtype=torch.long, device=device)

            pred_results = model.predict(bt, ei, ec, sos_id, eos_id, beam_width=beam_width)
            pred_tokens, score = pred_results[0]
            pred_name = sp.decode(pred_tokens).strip()

            results.append({
                "address": func_data["address"],
                "bap_name": func_name,
                "predicted_name": pred_name,
                "score": score / max(len(pred_tokens), 1) if pred_tokens else -999,
                "num_blocks": num_blocks,
                "num_ext_calls": len(ext),
            })

        return results

    def predict_all_models(
        self,
        binary_path: str,
        beam_width: int = 5,
        progress_callback=None,
    ) -> dict:
        if progress_callback:
            progress_callback("Running BAP to extract IR...")
        bir_path = self.lift_binary(binary_path)

        try:
            if progress_callback:
                progress_callback("Parsing functions from BAP-IR...")
            functions = self.parse_functions(bir_path)

            if progress_callback:
                progress_callback("Extracting external calls...")
            binary_name = os.path.splitext(os.path.basename(binary_path))[0]
            ext_by_func = self.extract_external_calls(bir_path, binary_name)

            predictions = {}
            dl_models = [n for n, m in self.manager.models.items() if m["type"] == "dl"]
            for i, model_name in enumerate(dl_models):
                if progress_callback:
                    progress_callback(f"Predicting with {model_name} ({i + 1}/{len(dl_models)})...")
                preds = self.predict_single_model(model_name, functions, ext_by_func, beam_width)
                predictions[model_name] = preds

            target_count = sum(
                1 for n, d in functions.items()
                if n.startswith("sub_") and d["num_blocks"] >= 2
            )

            return {
                "functions": functions,
                "predictions": predictions,
                "ext_by_func": ext_by_func,
                "stats": {
                    "total_functions": len(functions),
                    "target_functions": target_count,
                    "models_used": dl_models,
                },
            }
        finally:
            if os.path.exists(bir_path):
                os.remove(bir_path)

    def evaluate_with_ground_truth(
        self,
        predictions: Dict[str, List[dict]],
        ground_truth: Dict[str, str],
    ) -> Dict[str, dict]:
        evaluation = {}
        for model_name, preds in predictions.items():
            per_func = []
            for pred in preds:
                addr = pred["address"]
                if addr not in ground_truth:
                    continue
                true_name = ground_truth[addr]
                predicted = pred["predicted_name"]
                metrics = compute_all_metrics(predicted, true_name)
                per_func.append({
                    "address": addr,
                    "predicted": predicted,
                    "true_name": true_name,
                    **metrics,
                })

            if per_func:
                avg_metrics = {
                    "precision": sum(f["precision"] for f in per_func) / len(per_func),
                    "recall": sum(f["recall"] for f in per_func) / len(per_func),
                    "f1": sum(f["f1"] for f in per_func) / len(per_func),
                    "exact_match": sum(1 for f in per_func if f["exact_match"]) / len(per_func),
                    "char_ngram_sim": sum(f["char_ngram_sim"] for f in per_func) / len(per_func),
                    "edit_sim": sum(f["edit_sim"] for f in per_func) / len(per_func),
                    "num_evaluated": len(per_func),
                }
            else:
                avg_metrics = {
                    "precision": 0, "recall": 0, "f1": 0,
                    "exact_match": 0, "char_ngram_sim": 0, "edit_sim": 0,
                    "num_evaluated": 0,
                }

            evaluation[model_name] = {**avg_metrics, "per_function": per_func}

        return evaluation


def extract_ground_truth_from_debug_binary(debug_binary_path: str) -> Dict[str, str]:
    try:
        result = subprocess.run(
            ["nm", "--defined-only", "-C", debug_binary_path],
            capture_output=True, text=True, timeout=60,
        )
        gt = {}
        for line in result.stdout.strip().split("\n"):
            parts = line.split()
            if len(parts) >= 3:
                addr = "0x" + parts[0].lstrip("0").lower()
                if not addr.replace("0x", ""):
                    addr = "0x0"
                func_type = parts[1]
                name = parts[2]
                if func_type in ("T", "t"):
                    gt[addr] = name
        return gt
    except Exception as e:
        raise RuntimeError(f"Failed to extract ground truth with nm: {e}")
