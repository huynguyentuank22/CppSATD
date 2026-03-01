"""
run_grid.py
-----------
Full grid-search pipeline for SATD detection.
Loads data, builds stratified splits, trains all model variants, saves results.

All configuration is in CONFIG dict at the top of this file.
No argparse. Run directly: python run_grid.py
"""

from __future__ import annotations

import copy
import json
import logging
import os
import random
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

# ── local modules (same directory) ─────────────────────────────────────────
from models_baselines import (
    CommentOnlyClassifier,
    CodeOnlyClassifier,
    NUM_LABELS as BASELINE_NUM_LABELS,
    CONFIG as BASELINE_CONFIG,
)
from models_fusion import (
    DualEncoderLateFusion,
    DualEncoderCrossAttentionFusion,
    CONFIG as FUSION_CONFIG,
)
from train_eval import (
    set_seed,
    compute_class_weights,
    compute_metrics,
    run_training,
    evaluate,
    CONFIG as TRAIN_CONFIG,
)

warnings.filterwarnings("ignore")

# ===========================================================================
# MASTER CONFIG — edit these values; no argparse needed
# ===========================================================================
CONFIG: Dict[str, Any] = {
    # ── Paths ───────────────────────────────────────────────────────────────
    # Point DATA_PATH to the cleaned CSV (absolute or relative).
    # On Kaggle set to e.g. "/kaggle/input/cppsatd/manual_annotations_cleaned.csv"
    "DATA_PATH": "/kaggle/input/datasets/huy281204/cppsatd/manual_annotations_cleaned.csv",
    "OUTPUT_DIR": "/kaggle/working/satd_xai_outputs",

    # ── Dataset / Schema ────────────────────────────────────────────────────
    "DATASET_NAME": "cppsatd",                  # used only for logging

    # ── Labels ──────────────────────────────────────────────────────────────
    "LABELS": ["Non-SATD", "Design", "Requirement", "Defect", "Test", "Documentation"],
    # Canonical form used to match manual_annotation values (case-insensitive keys)
    "LABEL_ALIASES": {
        "non-satd": "Non-SATD",
        "design/code": "Design",
        "design": "Design",
        "requirement": "Requirement",
        "defect": "Defect",
        "test": "Test",
        "documentation": "Documentation",
    },

    # ── Split ────────────────────────────────────────────────────────────────
    "TRAIN_RATIO": 0.80,
    "VAL_RATIO":   0.10,
    "TEST_RATIO":  0.10,
    "SEED": 42,

    # ── Tokenisation ─────────────────────────────────────────────────────────
    "MAX_LEN_COMMENT": 128,
    "MAX_LEN_CODE":    384,

    # ── Training ─────────────────────────────────────────────────────────────
    "EPOCHS": 3,
    # P100 16 GB: batch_size=16 halves the number of steps → ~45% faster than batch=8
    # If you get OOM on late-fusion/cross-attn, drop back to 8.
    "BATCH_SIZE": 16,
    "WARMUP_RATIO": 0.06,
    "WEIGHT_DECAY": 0.01,
    "ES_PATIENCE": 2,
    "MAX_GRAD_NORM": 1.0,
    # P100 does NOT have Turing/Ampere Tensor Cores → AMP FP16 gives minimal speedup.
    # More importantly, DeBERTa-v3 stores some layers in FP16 internally, which causes
    # "Attempting to unscale FP16 gradients" crash with GradScaler. Keep USE_AMP=False.
    "USE_AMP": False,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",

    # ── LR grid ──────────────────────────────────────────────────────────────
    # Full grid = 3 LRs.  To fit in a ~9h Kaggle session use [2e-5] only,
    # then re-run with [1e-5, 5e-5] if you have time.
    "LR_GRID": [1e-5, 2e-5, 5e-5],

    # ── Models ───────────────────────────────────────────────────────────────
    "COMMENT_ENCODERS": [
        # "roberta-base",
        "microsoft/deberta-base",
        # "bert-base-uncased",
    ],
    "CODE_ENCODERS": [
        "microsoft/codebert-base",
        "microsoft/graphcodebert-base",
        # "microsoft/unixcoder-base-nine",
    ],

    # ── Grid control ─────────────────────────────────────────────────────────
    "TOP_K_FUSION_FOR_CROSSATTN": 3,    # how many late-fusion combos to promote to cross-attn
    "CROSS_ATTN_HEADS": 8,
    "CROSS_ATTN_PROJ_DIM": 768,

    # ── Phase control ──────────────────────────────────────────────────────────
    # Which phase to run.  0 = all phases sequentially (original behaviour).
    # 1 = comment-only baselines
    # 2 = code-only baselines
    # 3 = late-fusion dual encoder
    # 4 = cross-attention dual encoder  (reads results_phase3.json)
    # Overridden by  --phase N  CLI argument when calling: python run_grid.py --phase N
    "PHASE": 0,

    # ── Phase result paths (optional) ──────────────────────────────────────────
    # Only Phase 4 needs results from a previous phase:
    #   it reads Phase 3's results_phase3.json to select the top-K late-fusion
    #   combos before running cross-attention training.
    # Phases 1, 2, 3 are fully independent and need no prior results.
    #
    # If Phase 3 was run in a different Kaggle session, point PHASE3_RESULT_PATH
    # to the saved results_phase3.json (e.g. from a Kaggle Dataset input).
    # Leave as None to look for it automatically inside OUTPUT_DIR.
    #
    # Example (Kaggle):
    #   "/kaggle/input/satd-phase3-outputs/results_phase3.json"
    "PHASE3_RESULT_PATH": None,

    # ── Debug ────────────────────────────────────────────────────────────────
    # Set > 0 to subsample dataset for a quick smoke-test run;
    # set to 0 or None to use full data.
    "DEBUG_SUBSAMPLE": 0,
    # Set to True to only run a tiny grid (1 encoder each, 1 lr) — fast CI check
    "DEBUG_TINY_GRID": False,

    # ── P100 time-budget presets (uncomment the one you need) ─────────────────
    # PRESET A — smoke test (~20 min):
    #   "DEBUG_SUBSAMPLE": 2000, "DEBUG_TINY_GRID": True
    #
    # PRESET B — single-LR full encoders, no cross-attn (~5h on P100 batch=16):
    #   "LR_GRID": [2e-5], "TOP_K_FUSION_FOR_CROSSATTN": 0
    #
    # PRESET C — full grid, batch=16 (~12h, needs 2 Kaggle sessions or P100+):
    #   (default above)
}

NUM_LABELS: int = len(CONFIG["LABELS"])
LABEL2ID: Dict[str, int] = {lbl: i for i, lbl in enumerate(CONFIG["LABELS"])}
ID2LABEL: Dict[int, str] = {i: lbl for lbl, i in LABEL2ID.items()}

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ===========================================================================
# Data loading & normalisation
# ===========================================================================

def load_dataframe(path: str) -> pd.DataFrame:
    """Load CSV or Parquet; auto-detect by extension."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    ext = p.suffix.lower()
    if ext in (".parquet", ".pq"):
        df = pd.read_parquet(path)
        logger.info("Loaded parquet: %s  shape=%s", path, df.shape)
    else:
        df = pd.read_csv(path)
        logger.info("Loaded CSV: %s  shape=%s", path, df.shape)
    return df


def _coalesce(*vals: Any) -> str:
    """Return the first non-null, non-empty string."""
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply schema normalisation:
      - comment  : commenttext_normalized ??? commenttext_clean ??? commenttext
      - code     : preceding_code + "\n" + succeeding_code (either may be null)
      - label    : manual_annotation ??? LABEL_ALIASES ??? "Non-SATD"
      - id       : comment_id
      - project  : projectname
    """
    rows = []
    bad_label_count = 0

    for _, row in df.iterrows():
        # comment text
        comment = _coalesce(
            row.get("commenttext_normalized"),
            row.get("commenttext_clean"),
            row.get("commenttext"),
        )

        # code context
        pre  = _coalesce(row.get("preceding_code"))
        post = _coalesce(row.get("succeeding_code"))
        if pre and post:
            code = pre + "\n" + post
        else:
            code = pre or post

        # label
        raw_label = row.get("manual_annotation")
        if pd.isna(raw_label) or not isinstance(raw_label, str):
            label = "Non-SATD"
            bad_label_count += 1
        else:
            label = CONFIG["LABEL_ALIASES"].get(raw_label.strip().lower())
            if label is None:
                label = "Non-SATD"
                bad_label_count += 1

        rows.append({
            "comment_id":  str(row.get("comment_id", "")),
            "comment":     comment,
            "code":        code,
            "label":       label,
            "label_id":    LABEL2ID[label],
            "projectname": str(row.get("projectname", "")),
        })

    if bad_label_count:
        logger.warning("%d samples had unmappable labels and were set to 'Non-SATD'.", bad_label_count)

    out = pd.DataFrame(rows)
    logger.info("Normalised dataset: %d rows", len(out))
    logger.info("Label distribution:\n%s", out["label"].value_counts().to_string())
    return out


# ===========================================================================
# Stratified split (robust to tiny classes)
# ===========================================================================

def stratified_split(
    df: pd.DataFrame,
    train_ratio: float = CONFIG["TRAIN_RATIO"],
    val_ratio:   float = CONFIG["VAL_RATIO"],
    seed:        int   = CONFIG["SEED"],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    80 / 10 / 10 stratified split.
    For classes with < 3 samples, all samples go to train (cannot split them).
    Remaining samples are split proportionally.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    train_idx, val_idx, test_idx = [], [], []
    label_col = "label_id"

    for cls_id in range(NUM_LABELS):
        cls_indices = df.index[df[label_col] == cls_id].tolist()
        n = len(cls_indices)
        if n == 0:
            continue

        # Shuffle
        shuffled = cls_indices[:]
        rng.shuffle(shuffled)

        if n < 3:
            # Cannot split — put everything in train
            logger.warning(
                "Class %s has only %d sample(s); all sent to train.", ID2LABEL[cls_id], n
            )
            train_idx.extend(shuffled)
            continue

        n_val  = max(1, round(n * val_ratio))
        n_test = max(1, round(n * (1 - train_ratio - val_ratio)))  # same as val_ratio
        n_test = min(n_test, n - n_val - 1)          # ensure at least 1 in train
        n_train = n - n_val - n_test

        train_idx.extend(shuffled[:n_train])
        val_idx.extend(shuffled[n_train: n_train + n_val])
        test_idx.extend(shuffled[n_train + n_val:])

    # Shuffle index lists
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    train_df = df.loc[train_idx].reset_index(drop=True)
    val_df   = df.loc[val_idx].reset_index(drop=True)
    test_df  = df.loc[test_idx].reset_index(drop=True)

    def _dist(d: pd.DataFrame) -> str:
        return str(d["label"].value_counts().to_dict())

    logger.info("Train %d  |  Val %d  |  Test %d", len(train_df), len(val_df), len(test_df))
    logger.info("Train dist: %s", _dist(train_df))
    logger.info("Val   dist: %s", _dist(val_df))
    logger.info("Test  dist: %s", _dist(test_df))

    return train_df, val_df, test_df


# ===========================================================================
# PyTorch Dataset
# ===========================================================================

class SATDDataset(Dataset):
    """
    Returns tokenised comment + code inputs, label_id, comment_id.

    Stores offset mappings for XAI later.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        comment_tokenizer: Any,
        code_tokenizer: Any,
        max_len_comment: int = CONFIG["MAX_LEN_COMMENT"],
        max_len_code: int = CONFIG["MAX_LEN_CODE"],
        keep_offsets: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.c_tok = comment_tokenizer
        self.k_tok = code_tokenizer
        self.max_c = max_len_comment
        self.max_k = max_len_code
        self.keep_offsets = keep_offsets

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        comment_text = row["comment"] if row["comment"] else " "
        code_text    = row["code"]    if row["code"]    else " "

        c_enc = self.c_tok(
            comment_text,
            max_length=self.max_c,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=self.keep_offsets,
        )
        k_enc = self.k_tok(
            code_text,
            max_length=self.max_k,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=self.keep_offsets,
        )

        sample: Dict[str, Any] = {
            "comment_input_ids":      c_enc["input_ids"].squeeze(0),
            "comment_attention_mask": c_enc["attention_mask"].squeeze(0),
            "code_input_ids":         k_enc["input_ids"].squeeze(0),
            "code_attention_mask":    k_enc["attention_mask"].squeeze(0),
            "labels":                 torch.tensor(row["label_id"], dtype=torch.long),
            "comment_id":             row["comment_id"],
        }

        # token_type_ids (BERT family)
        if "token_type_ids" in c_enc:
            sample["comment_token_type_ids"] = c_enc["token_type_ids"].squeeze(0)
        if "token_type_ids" in k_enc:
            sample["code_token_type_ids"] = k_enc["token_type_ids"].squeeze(0)

        if self.keep_offsets:
            sample["comment_offset_mapping"] = c_enc["offset_mapping"].squeeze(0)
            sample["code_offset_mapping"] = k_enc["offset_mapping"].squeeze(0)
            sample["code_raw_text"] = code_text

        return sample


def _collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Custom collate that keeps string fields separate."""
    keys = batch[0].keys()
    out: Dict[str, Any] = {}
    for k in keys:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = vals   # list of strings / offsets
    return out


def build_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    comment_tokenizer: Any,
    code_tokenizer: Any,
    batch_size: int = CONFIG["BATCH_SIZE"],
    max_len_comment: int = CONFIG["MAX_LEN_COMMENT"],
    max_len_code: int = CONFIG["MAX_LEN_CODE"],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = SATDDataset(train_df, comment_tokenizer, code_tokenizer, max_len_comment, max_len_code)
    val_ds   = SATDDataset(val_df,   comment_tokenizer, code_tokenizer, max_len_comment, max_len_code)
    test_ds  = SATDDataset(test_df,  comment_tokenizer, code_tokenizer, max_len_comment, max_len_code)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=_collate_fn, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=_collate_fn, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              collate_fn=_collate_fn, num_workers=0)
    return train_loader, val_loader, test_loader


# ===========================================================================
# Tokenizer loader (cached)
# ===========================================================================

_TOK_CACHE: Dict[str, Any] = {}


def get_tokenizer(model_name: str) -> Any:
    if model_name not in _TOK_CACHE:
        logger.info("Loading tokenizer: %s", model_name)
        _TOK_CACHE[model_name] = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return _TOK_CACHE[model_name]


# ===========================================================================
# Wrapper: strip non-tensor keys before forward()
# ===========================================================================

class _ForwardWrapper(torch.nn.Module):
    """
    Wraps a model so that DataLoader batches (which may include comment_id
    strings, offset mapping lists, etc.) can be forwarded without KeyError.

    Also routes code/comment inputs for baseline models that only accept one side.
    """

    def __init__(self, model: torch.nn.Module, model_type: str) -> None:
        super().__init__()
        self.model = model
        self.model_type = model_type  # "comment", "code", "late_fusion", "cross_attn"

    def forward(self, **kwargs: Any) -> Dict[str, torch.Tensor]:
        # Keep only tensor kwargs accepted by the underlying model
        _COMMENT_KEYS = {"comment_input_ids", "comment_attention_mask", "comment_token_type_ids", "labels"}
        _CODE_KEYS    = {"code_input_ids",    "code_attention_mask",    "code_token_type_ids",    "labels"}
        _FUSION_KEYS  = _COMMENT_KEYS | _CODE_KEYS

        if self.model_type == "comment":
            valid = {k: v for k, v in kwargs.items() if k in _COMMENT_KEYS and isinstance(v, torch.Tensor)}
        elif self.model_type == "code":
            valid = {k: v for k, v in kwargs.items() if k in _CODE_KEYS and isinstance(v, torch.Tensor)}
        else:
            valid = {k: v for k, v in kwargs.items() if k in _FUSION_KEYS and isinstance(v, torch.Tensor)}

        return self.model(**valid)

    def parameters(self, recurse: bool = True):
        return self.model.parameters(recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):
        return self.model.named_parameters(prefix, recurse)

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.model.load_state_dict(*args, **kwargs)

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def eval(self):
        self.model.eval()
        return self

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        return self


# ===========================================================================
# Single run helper
# ===========================================================================

def run_single(
    run_cfg: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Build, train, and evaluate one model configuration.
    Returns a result dict.
    """
    model_type = run_cfg["model_type"]
    lr = run_cfg["lr"]

    set_seed(CONFIG["SEED"])

    if model_type == "comment":
        base_model = CommentOnlyClassifier(
            comment_encoder_name=run_cfg["comment_encoder"],
            num_labels=NUM_LABELS,
            class_weights=class_weights,
        )
    elif model_type == "code":
        base_model = CodeOnlyClassifier(
            code_encoder_name=run_cfg["code_encoder"],
            num_labels=NUM_LABELS,
            class_weights=class_weights,
        )
    elif model_type == "late_fusion":
        base_model = DualEncoderLateFusion(
            comment_model_name=run_cfg["comment_encoder"],
            code_model_name=run_cfg["code_encoder"],
            num_labels=NUM_LABELS,
            class_weights=class_weights,
        )
    elif model_type == "cross_attn":
        base_model = DualEncoderCrossAttentionFusion(
            comment_model_name=run_cfg["comment_encoder"],
            code_model_name=run_cfg["code_encoder"],
            num_labels=NUM_LABELS,
            heads=CONFIG["CROSS_ATTN_HEADS"],
            proj_dim=CONFIG["CROSS_ATTN_PROJ_DIM"],
            class_weights=class_weights,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    wrapped = _ForwardWrapper(base_model, model_type)

    train_result = run_training(
        model=wrapped,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=CONFIG["EPOCHS"],
        lr=lr,
        warmup_ratio=CONFIG["WARMUP_RATIO"],
        weight_decay=CONFIG["WEIGHT_DECAY"],
        es_patience=CONFIG["ES_PATIENCE"],
        use_amp=CONFIG["USE_AMP"],
        num_labels=NUM_LABELS,
        label_names=CONFIG["LABELS"],
        max_grad_norm=CONFIG["MAX_GRAD_NORM"],
    )

    # Reload best weights for test evaluation
    wrapped.model.to("cpu")
    wrapped.model.load_state_dict(train_result["best_state_dict"])
    wrapped.model.to(device)

    _, test_preds, test_labels = evaluate(wrapped, test_loader, device)
    test_metrics = compute_metrics(
        test_labels.tolist(), test_preds.tolist(), NUM_LABELS, CONFIG["LABELS"]
    )

    result = {
        "model_type":        model_type,
        "comment_encoder":   run_cfg.get("comment_encoder", ""),
        "code_encoder":      run_cfg.get("code_encoder", ""),
        "lr":                lr,
        "best_val_macro_f1": train_result["best_val_macro_f1"],
        "val_metrics":       train_result["best_val_metrics"],
        "test_macro_f1":     test_metrics["macro_f1"],
        "test_metrics":      test_metrics,
        "best_epoch":        train_result["best_epoch"],
        "history":           train_result["history"],
        "state_dict":        train_result["best_state_dict"],
    }
    logger.info(
        "[RESULT] type=%-12s  comment=%-35s  code=%-40s  lr=%.0e  val_f1=%.4f  test_f1=%.4f",
        model_type,
        run_cfg.get("comment_encoder", "-"),
        run_cfg.get("code_encoder", "-"),
        lr,
        result["best_val_macro_f1"],
        result["test_macro_f1"],
    )
    return result


# ===========================================================================
# Phase result helpers
# ===========================================================================

def _save_phase_results(results: List[Dict], phase: int, output_dir: Path) -> None:
    """
    Persist results from a single phase.
    Saves:
      results_phase<N>.json  — full metrics (no state_dicts)
      results_phase<N>.csv   — flat summary table
    """
    # JSON (drop state_dicts — too large)
    json_results = [{k: v for k, v in r.items() if k != "state_dict"} for r in results]
    json_path = output_dir / f"results_phase{phase}.json"
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    logger.info("Phase %d → saved %s", phase, json_path)

    # CSV
    rows_csv = []
    for r in results:
        row: Dict[str, Any] = {
            "model_type":        r["model_type"],
            "comment_encoder":   r.get("comment_encoder", ""),
            "code_encoder":      r.get("code_encoder", ""),
            "lr":                r["lr"],
            "best_val_macro_f1": r["best_val_macro_f1"],
            "test_macro_f1":     r["test_macro_f1"],
            "best_epoch":        r.get("best_epoch"),
        }
        rows_csv.append(row)
    csv_path = output_dir / f"results_phase{phase}.csv"
    pd.DataFrame(rows_csv).to_csv(csv_path, index=False)
    logger.info("Phase %d → saved %s", phase, csv_path)


def _load_phase_results(phase: int, output_dir: Path) -> List[Dict]:
    """Load a previously saved phase result JSON (no state_dicts).

    For phase 3 specifically, resolution order is:
      1. CONFIG["PHASE3_RESULT_PATH"]       — explicit override path
      2. output_dir / results_phase3.json   — default location
    All other phases use only the default location.
    """
    # Explicit override only supported for phase 3
    if phase == 3:
        override = CONFIG.get("PHASE3_RESULT_PATH")
        if override:
            path = Path(override)
            if not path.exists():
                raise FileNotFoundError(
                    f"PHASE3_RESULT_PATH points to a non-existent file: {path}"
                )
            with open(path) as f:
                data = json.load(f)
            logger.info("Loaded phase %d results (%d runs) from %s", phase, len(data), path)
            return data

    # Default location
    path = output_dir / f"results_phase{phase}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Phase {phase} results not found at {path}.\n"
            f"Either run  python run_grid.py --phase {phase}  first"
            + (", or set CONFIG['PHASE3_RESULT_PATH'] to the correct path." if phase == 3 else ".")
        )
    with open(path) as f:
        data = json.load(f)
    logger.info("Loaded phase %d results (%d runs) from %s", phase, len(data), path)
    return data


def _save_grid_summary(all_results: List[Dict], output_dir: Path) -> None:
    """Save the combined results_grid.csv / results_grid.json and best_model/."""
    rows_csv = []
    for r in all_results:
        row: Dict[str, Any] = {
            "model_type":        r["model_type"],
            "comment_encoder":   r.get("comment_encoder", ""),
            "code_encoder":      r.get("code_encoder", ""),
            "lr":                r["lr"],
            "best_val_macro_f1": r["best_val_macro_f1"],
            "test_macro_f1":     r["test_macro_f1"],
            "best_epoch":        r.get("best_epoch"),
        }
        rows_csv.append(row)
    results_df = pd.DataFrame(rows_csv)
    results_df.to_csv(output_dir / "results_grid.csv", index=False)
    logger.info("Saved results_grid.csv")

    json_results = [{k: v for k, v in r.items() if k != "state_dict"} for r in all_results]
    with open(output_dir / "results_grid.json", "w") as f:
        json.dump(json_results, f, indent=2)
    logger.info("Saved results_grid.json")

    # Best model by val macro f1
    best_result = max(all_results, key=lambda r: r["best_val_macro_f1"])
    logger.info(
        "\n*** BEST RUN ***\n  type=%s  comment=%s  code=%s  lr=%.0e  val_f1=%.4f  test_f1=%.4f",
        best_result["model_type"],
        best_result.get("comment_encoder", "-"),
        best_result.get("code_encoder", "-"),
        best_result["lr"],
        best_result["best_val_macro_f1"],
        best_result["test_macro_f1"],
    )

    # Only save checkpoint if state_dict is present (not available when loaded from JSON)
    best_dir = output_dir / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)

    if best_result.get("state_dict") is not None:
        torch.save(best_result["state_dict"], best_dir / "model.pt")
        logger.info("Saved best checkpoint to %s", best_dir / "model.pt")
    else:
        logger.warning(
            "state_dict not available (loaded from JSON). "
            "Rerun the winning phase to get model.pt."
        )

    best_config = {
        "model_type":        best_result["model_type"],
        "comment_encoder":   best_result.get("comment_encoder", ""),
        "code_encoder":      best_result.get("code_encoder", ""),
        "lr":                best_result["lr"],
        "max_len_comment":   CONFIG["MAX_LEN_COMMENT"],
        "max_len_code":      CONFIG["MAX_LEN_CODE"],
        "num_labels":        NUM_LABELS,
        "label2id":          LABEL2ID,
        "id2label":          {str(k): v for k, v in ID2LABEL.items()},
        "labels":            CONFIG["LABELS"],
        "best_val_macro_f1": best_result["best_val_macro_f1"],
        "test_macro_f1":     best_result["test_macro_f1"],
        "best_epoch":        best_result.get("best_epoch"),
        "val_metrics":       best_result.get("val_metrics"),
        "test_metrics":      best_result.get("test_metrics"),
        "split_info_path":   str(output_dir / "split_info.json"),
        "seed":              CONFIG["SEED"],
        "cross_attn_heads":    CONFIG["CROSS_ATTN_HEADS"],
        "cross_attn_proj_dim": CONFIG["CROSS_ATTN_PROJ_DIM"],
        "pooling":             "mean",
        "dropout":             0.1,
    }
    with open(best_dir / "best_config.json", "w") as f:
        json.dump(best_config, f, indent=2)
    logger.info("Saved best_config.json to %s", best_dir / "best_config.json")


# ===========================================================================
# Main grid-search driver
# ===========================================================================

def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(
        description="Grid-search pipeline for SATD detection."
    )
    ap.add_argument(
        "--phase",
        type=int,
        default=CONFIG.get("PHASE", 0),
        choices=[0, 1, 2, 3, 4],
        help=(
            "Phase to run (default: %(default)s).\n"
            "  0 = all phases sequentially\n"
            "  1 = comment-only baselines       (~3.5h on Kaggle P100)\n"
            "  2 = code-only baselines          (~3.5h on Kaggle P100)\n"
            "  3 = late-fusion dual encoder     (~9h  on Kaggle P100)\n"
            "  4 = cross-attention dual encoder (~3h  on Kaggle P100)\n"
            "          Phase 4 requires results_phase3.json to exist first."
        ),
    )
    args, _unknown = ap.parse_known_args()  # _unknown ignores Jupyter/papermill extras
    phase = args.phase

    logger.info("=" * 70)
    logger.info("run_grid  PHASE=%d%s", phase, "  (all)" if phase == 0 else "")
    logger.info("=" * 70)

    set_seed(CONFIG["SEED"])
    device = torch.device(CONFIG["DEVICE"])
    output_dir = Path(CONFIG["OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 4 only: load phase 3 results, skip data setup if desired ─────
    # (we still need data for training, so setup always runs)

    # ── 1. Load & normalise data ────────────────────────────────────────────
    df_raw = load_dataframe(CONFIG["DATA_PATH"])
    df = normalize_dataset(df_raw)

    if CONFIG.get("DEBUG_SUBSAMPLE") and CONFIG["DEBUG_SUBSAMPLE"] > 0:
        n_sub = int(CONFIG["DEBUG_SUBSAMPLE"])
        logger.warning("DEBUG_SUBSAMPLE=%d — using only %d rows!", n_sub, n_sub)
        df = df.sample(n=min(n_sub, len(df)), random_state=CONFIG["SEED"]).reset_index(drop=True)

    # ── 2. Stratified split ─────────────────────────────────────────────────
    train_df, val_df, test_df = stratified_split(
        df,
        train_ratio=CONFIG["TRAIN_RATIO"],
        val_ratio=CONFIG["VAL_RATIO"],
        seed=CONFIG["SEED"],
    )

    # Save split indices for XAI reproducibility (idempotent overwrite)
    split_info: Dict[str, Any] = {
        "train_comment_ids": train_df["comment_id"].tolist(),
        "val_comment_ids":   val_df["comment_id"].tolist(),
        "test_comment_ids":  test_df["comment_id"].tolist(),
        "train_distribution": train_df["label"].value_counts().to_dict(),
        "val_distribution":   val_df["label"].value_counts().to_dict(),
        "test_distribution":  test_df["label"].value_counts().to_dict(),
        "label2id": LABEL2ID,
        "id2label": {str(k): v for k, v in ID2LABEL.items()},
    }
    with open(output_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)
    logger.info("Split info saved to %s", output_dir / "split_info.json")

    # ── 3. Class weights from train ─────────────────────────────────────────
    class_weights = compute_class_weights(train_df["label_id"].tolist(), NUM_LABELS)

    # ── 4. Grid setup ───────────────────────────────────────────────────────
    lr_grid          = CONFIG["LR_GRID"]
    comment_encoders = CONFIG["COMMENT_ENCODERS"]
    code_encoders    = CONFIG["CODE_ENCODERS"]

    if CONFIG.get("DEBUG_TINY_GRID"):
        logger.warning("DEBUG_TINY_GRID=True — trimming grid to 1 encoder each and 1 LR.")
        comment_encoders = comment_encoders[:1]
        code_encoders    = code_encoders[:1]
        lr_grid          = lr_grid[:1]

    all_results: List[Dict] = []

    # ── helper: run with given tokenizers ──────────────────────────────────
    def _run_combo(run_cfg: Dict, c_tok_name: str, k_tok_name: str) -> Dict:
        c_tok = get_tokenizer(c_tok_name)
        k_tok = get_tokenizer(k_tok_name)
        tr_l, va_l, te_l = build_loaders(
            train_df, val_df, test_df,
            comment_tokenizer=c_tok,
            code_tokenizer=k_tok,
            batch_size=CONFIG["BATCH_SIZE"],
            max_len_comment=CONFIG["MAX_LEN_COMMENT"],
            max_len_code=CONFIG["MAX_LEN_CODE"],
        )
        return run_single(run_cfg, tr_l, va_l, te_l, class_weights, device, output_dir)

    # =========================================================================
    # PHASE 1 — Comment-only baselines
    # =========================================================================
    if phase in (0, 1):
        logger.info("=" * 70)
        logger.info("PHASE 1: Comment-only baselines")
        logger.info("=" * 70)
        phase1_results: List[Dict] = []
        for c_enc in comment_encoders:
            for lr in lr_grid:
                run_cfg = {"model_type": "comment", "comment_encoder": c_enc, "lr": lr}
                res = _run_combo(run_cfg, c_tok_name=c_enc, k_tok_name=c_enc)
                phase1_results.append(res)
                all_results.append(res)
        _save_phase_results(phase1_results, 1, output_dir)
        if phase == 1:
            logger.info("Phase 1 complete. Outputs at: %s", output_dir)
            return

    # =========================================================================
    # PHASE 2 — Code-only baselines
    # =========================================================================
    if phase in (0, 2):
        logger.info("=" * 70)
        logger.info("PHASE 2: Code-only baselines")
        logger.info("=" * 70)
        phase2_results: List[Dict] = []
        for k_enc in code_encoders:
            for lr in lr_grid:
                run_cfg = {"model_type": "code", "code_encoder": k_enc, "lr": lr}
                res = _run_combo(run_cfg, c_tok_name=k_enc, k_tok_name=k_enc)
                phase2_results.append(res)
                all_results.append(res)
        _save_phase_results(phase2_results, 2, output_dir)
        if phase == 2:
            logger.info("Phase 2 complete. Outputs at: %s", output_dir)
            return

    # =========================================================================
    # PHASE 3 — Late-fusion dual encoder  (9 combos × 3 LRs)
    # =========================================================================
    if phase in (0, 3):
        logger.info("=" * 70)
        logger.info("PHASE 3: Late-fusion dual encoder")
        logger.info("=" * 70)
        phase3_results: List[Dict] = []
        for c_enc in comment_encoders:
            for k_enc in code_encoders:
                for lr in lr_grid:
                    run_cfg = {
                        "model_type":      "late_fusion",
                        "comment_encoder": c_enc,
                        "code_encoder":    k_enc,
                        "lr":              lr,
                    }
                    res = _run_combo(run_cfg, c_tok_name=c_enc, k_tok_name=k_enc)
                    phase3_results.append(res)
                    all_results.append(res)
        _save_phase_results(phase3_results, 3, output_dir)
        if phase == 3:
            logger.info("Phase 3 complete. Outputs at: %s", output_dir)
            return

    # =========================================================================
    # PHASE 4 — Cross-attention dual encoder  (top-K combos × 3 LRs)
    # =========================================================================
    if phase in (0, 4):
        logger.info("=" * 70)
        logger.info("PHASE 4: Cross-attention dual encoder")
        logger.info("=" * 70)

        # When running phase 4 standalone, read phase 3 results from disk
        if phase == 4:
            lf_results = _load_phase_results(3, output_dir)
        else:
            lf_results = [r for r in all_results if r["model_type"] == "late_fusion"]

        # Select top-K combos by best val_f1 across all LRs
        combo_best: Dict[Tuple[str, str], float] = {}
        for r in lf_results:
            key = (r["comment_encoder"], r["code_encoder"])
            if r["best_val_macro_f1"] > combo_best.get(key, -1.0):
                combo_best[key] = r["best_val_macro_f1"]

        top_k = CONFIG["TOP_K_FUSION_FOR_CROSSATTN"]
        top_combos = sorted(combo_best.items(), key=lambda x: x[1], reverse=True)[:top_k]
        logger.info("Top-%d late-fusion combos selected for cross-attn:", top_k)
        for (c, k), f1 in top_combos:
            logger.info("  comment=%-35s  code=%-40s  val_f1=%.4f", c, k, f1)

        phase4_results: List[Dict] = []
        for (c_enc, k_enc), _ in top_combos:
            for lr in lr_grid:
                run_cfg = {
                    "model_type":      "cross_attn",
                    "comment_encoder": c_enc,
                    "code_encoder":    k_enc,
                    "lr":              lr,
                }
                res = _run_combo(run_cfg, c_tok_name=c_enc, k_tok_name=k_enc)
                phase4_results.append(res)
                all_results.append(res)
        _save_phase_results(phase4_results, 4, output_dir)

        # After phase 4: aggregate ALL phases into results_grid.*
        if phase == 4:
            logger.info("Phase 4 complete — aggregating all phases into results_grid.*")
            all_results = []
            for p in (1, 2, 3, 4):
                try:
                    all_results.extend(_load_phase_results(p, output_dir))
                except FileNotFoundError as e:
                    logger.warning("Skipping missing phase: %s", e)
            all_results.extend(phase4_results)  # use in-memory copy (has state_dict)
            # de-duplicate: prefer in-memory entry (has state_dict) over JSON entry
            seen: set = set()
            deduped: List[Dict] = []
            for r in reversed(all_results):
                key = (
                    r["model_type"],
                    r.get("comment_encoder", ""),
                    r.get("code_encoder", ""),
                    r["lr"],
                )
                if key not in seen:
                    seen.add(key)
                    deduped.append(r)
            all_results = list(reversed(deduped))

        _save_grid_summary(all_results, output_dir)
        logger.info("Done. Outputs at: %s", output_dir)
        return

    # phase == 0: all phases ran above, save combined summary
    logger.info("=" * 70)
    logger.info("All phases complete — saving combined results_grid.*")
    logger.info("=" * 70)
    _save_grid_summary(all_results, output_dir)
    logger.info("Done. Outputs at: %s", output_dir)


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    main()
