"""
explain_faithfulness.py
-----------------------
XAI + Faithfulness evaluation for the best SATD detection model.

Loads:
  /kaggle/working/satd_xai_outputs/best_model/best_config.json
  /kaggle/working/satd_xai_outputs/best_model/model.pt
  /kaggle/working/satd_xai_outputs/split_info.json
  /kaggle/input/cppsatd/manual_annotations_cleaned.csv  (original data)

Produces:
  /kaggle/working/satd_xai_outputs/faithfulness_summary.json
  /kaggle/working/satd_xai_outputs/per_sample_explanations.jsonl

All configuration in CONFIG dict; no argparse.
Self-contained — does not import from other project files.
"""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Auto-install captum if missing
# ---------------------------------------------------------------------------
try:
    from captum.attr import LayerIntegratedGradients
    CAPTUM_OK = True
except ImportError:
    logger_pre = logging.getLogger(__name__)
    logger_pre.warning("captum not found — installing …")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "captum", "-q"])
    from captum.attr import LayerIntegratedGradients  # type: ignore
    CAPTUM_OK = True

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CONFIG: Dict[str, Any] = {
    # Paths
    "OUTPUT_DIR":    "/kaggle/working/satd_xai_outputs",
    "BEST_MODEL_DIR": "/kaggle/working/satd_xai_outputs/best_model",
    # Original data (CSV or Parquet)
    "DATA_PATH": "/kaggle/input/cppsatd/manual_annotations_cleaned.csv",

    # XAI subset size (use all test if test is smaller)
    "XAI_SUBSET_SIZE": 500,
    # Faithfulness k values
    "FAITHFULNESS_K": [5, 10, 20],
    # Top tokens kept for evidence reporting
    "TOP_TOKENS_REPORT": 10,

    # Device
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    # Batch for model forward during IG
    "IG_N_STEPS": 50,
    # Internal batch for IG (lower = less memory)
    "IG_INTERNAL_BATCH": 8,

    # Label aliases (same as run_grid)
    "LABEL_ALIASES": {
        "non-satd":   "Non-SATD",
        "design/code": "Design",
        "design":     "Design",
        "requirement": "Requirement",
        "defect":     "Defect",
        "test":       "Test",
        "documentation": "Documentation",
    },
    "LABELS": ["Non-SATD", "Design", "Requirement", "Defect", "Test", "Documentation"],
}

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ===========================================================================
# Shared utilities (self-contained copies)
# ===========================================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _coalesce(*vals: Any) -> str:
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def cls_pool(token_embeddings: torch.Tensor, _attention_mask: torch.Tensor) -> torch.Tensor:
    return token_embeddings[:, 0, :]


# ===========================================================================
# Rebuild model classes (self-contained, no import from other files)
# ===========================================================================

class _Head(nn.Module):
    def __init__(self, in_dim: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_dim, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.drop(x))


class _CommentOnlyModel(nn.Module):
    def __init__(self, enc_name: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(enc_name)
        self.head = _Head(self.encoder.config.hidden_size, num_labels, dropout)

    def forward(self, input_ids, attention_mask, token_type_ids=None, **__):
        kw = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            kw["token_type_ids"] = token_type_ids
        out = self.encoder(**kw)
        pooled = mean_pool(out.last_hidden_state, attention_mask)
        return self.head(pooled)

    def get_word_embeddings(self):
        return self.encoder.embeddings.word_embeddings


class _CodeOnlyModel(nn.Module):
    def __init__(self, enc_name: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(enc_name)
        self.head = _Head(self.encoder.config.hidden_size, num_labels, dropout)

    def forward(self, input_ids, attention_mask, token_type_ids=None, **__):
        kw = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            kw["token_type_ids"] = token_type_ids
        out = self.encoder(**kw)
        pooled = mean_pool(out.last_hidden_state, attention_mask)
        return self.head(pooled)

    def get_word_embeddings(self):
        return self.encoder.embeddings.word_embeddings


class _LateFusionModel(nn.Module):
    def __init__(self, c_name: str, k_name: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.c_enc = AutoModel.from_pretrained(c_name)
        self.k_enc = AutoModel.from_pretrained(k_name)
        c_dim = self.c_enc.config.hidden_size
        k_dim = self.k_enc.config.hidden_size
        if c_dim != k_dim:
            target = max(c_dim, k_dim)
            self.c_proj = nn.Linear(c_dim, target) if c_dim != target else None
            self.k_proj = nn.Linear(k_dim, target) if k_dim != target else None
            fusion_dim = target
        else:
            self.c_proj = None
            self.k_proj = None
            fusion_dim = c_dim
        self.head = _Head(2 * fusion_dim, num_labels, dropout)

    def _enc_comment(self, iids, amask, ttids=None):
        kw = dict(input_ids=iids, attention_mask=amask)
        if ttids is not None:
            kw["token_type_ids"] = ttids
        out = self.c_enc(**kw)
        rep = mean_pool(out.last_hidden_state, amask)
        return self.c_proj(rep) if self.c_proj else rep

    def _enc_code(self, iids, amask, ttids=None):
        kw = dict(input_ids=iids, attention_mask=amask)
        if ttids is not None:
            kw["token_type_ids"] = ttids
        out = self.k_enc(**kw)
        rep = mean_pool(out.last_hidden_state, amask)
        return self.k_proj(rep) if self.k_proj else rep

    def forward(self,
                comment_input_ids, comment_attention_mask,
                code_input_ids, code_attention_mask,
                comment_token_type_ids=None, code_token_type_ids=None, **__):
        c = self._enc_comment(comment_input_ids, comment_attention_mask, comment_token_type_ids)
        k = self._enc_code(code_input_ids, code_attention_mask, code_token_type_ids)
        return self.head(torch.cat([c, k], dim=-1))

    def get_comment_word_embeddings(self):
        return self.c_enc.embeddings.word_embeddings

    def get_code_word_embeddings(self):
        return self.k_enc.embeddings.word_embeddings


class _CrossAttnModel(nn.Module):
    def __init__(self, c_name: str, k_name: str, num_labels: int,
                 heads: int = 8, proj_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.c_enc = AutoModel.from_pretrained(c_name)
        self.k_enc = AutoModel.from_pretrained(k_name)
        c_dim = self.c_enc.config.hidden_size
        k_dim = self.k_enc.config.hidden_size
        self.c_proj = nn.Linear(c_dim, proj_dim) if c_dim != proj_dim else nn.Identity()
        self.k_proj = nn.Linear(k_dim, proj_dim) if k_dim != proj_dim else nn.Identity()
        safe_heads = heads
        while proj_dim % safe_heads != 0 and safe_heads > 1:
            safe_heads -= 1
        self.cross_attn = nn.MultiheadAttention(proj_dim, safe_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(proj_dim)
        self.drop = nn.Dropout(dropout)
        self.head = _Head(2 * proj_dim, num_labels, dropout)

    def forward(self,
                comment_input_ids, comment_attention_mask,
                code_input_ids, code_attention_mask,
                comment_token_type_ids=None, code_token_type_ids=None, **__):
        ck = dict(input_ids=comment_input_ids, attention_mask=comment_attention_mask)
        if comment_token_type_ids is not None:
            ck["token_type_ids"] = comment_token_type_ids
        kk = dict(input_ids=code_input_ids, attention_mask=code_attention_mask)
        if code_token_type_ids is not None:
            kk["token_type_ids"] = code_token_type_ids

        c_h = self.c_proj(self.c_enc(**ck).last_hidden_state)
        k_h = self.k_proj(self.k_enc(**kk).last_hidden_state)
        kpm = (code_attention_mask == 0)
        attn_out, _ = self.cross_attn(c_h, k_h, k_h, key_padding_mask=kpm)
        c_att = self.norm(c_h + self.drop(attn_out))
        c_rep = mean_pool(c_att, comment_attention_mask)
        k_rep = mean_pool(k_h, code_attention_mask)
        return self.head(torch.cat([c_rep, k_rep], dim=-1))

    def get_comment_word_embeddings(self):
        return self.c_enc.embeddings.word_embeddings

    def get_code_word_embeddings(self):
        return self.k_enc.embeddings.word_embeddings


# ===========================================================================
# Model builder
# ===========================================================================

def build_model(cfg: Dict) -> nn.Module:
    """Instantiate the correct model class from best_config.json."""
    mtype = cfg["model_type"]
    nl = cfg["num_labels"]
    if mtype == "comment":
        return _CommentOnlyModel(cfg["comment_encoder"], nl)
    elif mtype == "code":
        return _CodeOnlyModel(cfg["code_encoder"], nl)
    elif mtype == "late_fusion":
        return _LateFusionModel(cfg["comment_encoder"], cfg["code_encoder"], nl)
    elif mtype == "cross_attn":
        return _CrossAttnModel(
            cfg["comment_encoder"], cfg["code_encoder"], nl,
            heads=cfg.get("cross_attn_heads", 8),
            proj_dim=cfg.get("cross_attn_proj_dim", 768),
        )
    else:
        raise ValueError(f"Unknown model_type: {mtype!r}")


# ===========================================================================
# Data helpers
# ===========================================================================

def load_dataframe(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_parquet(path) if p.suffix.lower() in (".parquet", ".pq") else pd.read_csv(path)


def normalize_sample(row: pd.Series, label_aliases: Dict, label2id: Dict) -> Optional[Dict]:
    comment = _coalesce(
        row.get("commenttext_normalized"),
        row.get("commenttext_clean"),
        row.get("commenttext"),
    )
    pre  = _coalesce(row.get("preceding_code"))
    post = _coalesce(row.get("succeeding_code"))
    code = (pre + "\n" + post) if (pre and post) else (pre or post)

    raw = row.get("manual_annotation")
    if pd.isna(raw) or not isinstance(raw, str):
        label = "Non-SATD"
    else:
        label = label_aliases.get(raw.strip().lower(), "Non-SATD")
    return {
        "comment_id": str(row.get("comment_id", "")),
        "comment":    comment or " ",
        "code":       code or " ",
        "label":      label,
        "label_id":   label2id[label],
    }


def stratified_subset(df: pd.DataFrame, n: int, label_col: str = "label_id",
                      num_labels: int = 6, seed: int = 42) -> pd.DataFrame:
    """Draw a stratified sample of size n (or all if smaller)."""
    if len(df) <= n:
        return df.reset_index(drop=True)
    rng = random.Random(seed)
    # proportional allocation
    label_counts = df[label_col].value_counts().to_dict()
    total = len(df)
    allocated: Dict[int, int] = {}
    for lbl, cnt in label_counts.items():
        allocated[lbl] = max(1, round(n * cnt / total))
    # correct sum
    while sum(allocated.values()) > n:
        biggest = max(allocated, key=lambda l: allocated[l])
        allocated[biggest] -= 1
    while sum(allocated.values()) < n:
        smallest_ratio = min(allocated, key=lambda l: allocated[l] / label_counts.get(l, 1))
        allocated[smallest_ratio] += 1

    chosen = []
    for lbl, k in allocated.items():
        idxs = df.index[df[label_col] == lbl].tolist()
        rng.shuffle(idxs)
        chosen.extend(idxs[:k])
    return df.loc[chosen].reset_index(drop=True)


# ===========================================================================
# Tokeniser helper
# ===========================================================================

def get_tokenizer(name: str):
    return AutoTokenizer.from_pretrained(name, use_fast=True)


def tokenize_single(tokenizer, text: str, max_len: int, device: torch.device):
    enc = tokenizer(
        text,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    return {k: v.to(device) for k, v in enc.items()}


# ===========================================================================
# IG attribution
# ===========================================================================

def _make_forward_fn_comment_only(model: _CommentOnlyModel, attention_mask: torch.Tensor,
                                   token_type_ids: Optional[torch.Tensor] = None):
    """Return a forward function: embedding_output -> logits (for LIG)."""
    def forward_fn(inputs_embeds: torch.Tensor) -> torch.Tensor:
        # We need to pass embeddings directly to BERT-style encoder
        enc_kw: Dict = dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        if token_type_ids is not None:
            enc_kw["token_type_ids"] = token_type_ids
        out = model.encoder(**enc_kw)
        pooled = mean_pool(out.last_hidden_state, attention_mask)
        return model.head(pooled)
    return forward_fn


def _make_forward_fn_code_only(model: _CodeOnlyModel, attention_mask: torch.Tensor,
                                token_type_ids: Optional[torch.Tensor] = None):
    def forward_fn(inputs_embeds: torch.Tensor) -> torch.Tensor:
        enc_kw: Dict = dict(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        if token_type_ids is not None:
            enc_kw["token_type_ids"] = token_type_ids
        out = model.encoder(**enc_kw)
        pooled = mean_pool(out.last_hidden_state, attention_mask)
        return model.head(pooled)
    return forward_fn


def _embed(model_encoder, input_ids: torch.Tensor) -> torch.Tensor:
    """Get word embeddings for given input_ids."""
    try:
        return model_encoder.embeddings.word_embeddings(input_ids)
    except AttributeError:
        return model_encoder.embeddings.word_embeddings(input_ids)


def compute_ig_attributions(
    model: nn.Module,
    model_type: str,
    comment_enc,
    code_enc,
    comment_tk_ids: torch.Tensor,     # (1, Lc)
    comment_attn_mask: torch.Tensor,
    code_tk_ids: torch.Tensor,         # (1, Lk)
    code_attn_mask: torch.Tensor,
    target_class: int,
    n_steps: int = CONFIG["IG_N_STEPS"],
    internal_batch: int = CONFIG["IG_INTERNAL_BATCH"],
    comment_ttids: Optional[torch.Tensor] = None,
    code_ttids: Optional[torch.Tensor] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Run Integrated Gradients.

    Returns:
        comment_attrs: (Lc,) abs summed attributions or None
        code_attrs:    (Lk,) abs summed attributions or None
    """
    model.eval()

    comment_attrs = None
    code_attrs = None

    if model_type in ("comment", "code"):
        is_comment = model_type == "comment"
        enc = model.encoder if is_comment else model.encoder
        attn = comment_attn_mask if is_comment else code_attn_mask
        ttids = comment_ttids if is_comment else code_ttids
        tk_ids = comment_tk_ids if is_comment else code_tk_ids

        if is_comment:
            fwd_fn = _make_forward_fn_comment_only(model, attn, ttids)
        else:
            fwd_fn = _make_forward_fn_code_only(model, attn, ttids)

        emb_layer = enc.embeddings.word_embeddings
        embeds = emb_layer(tk_ids)                        # (1, L, H)
        baselines = torch.zeros_like(embeds)

        lig = LayerIntegratedGradients(
            lambda emb: fwd_fn(emb),
            None,   # no layer needed — we pass embeddings directly
        )
        # Use standard IntegratedGradients on embeddings
        from captum.attr import IntegratedGradients
        ig = IntegratedGradients(lambda emb: fwd_fn(emb))
        attrs, _ = ig.attribute(
            inputs=embeds,
            baselines=baselines,
            target=target_class,
            n_steps=n_steps,
            internal_batch_size=internal_batch,
            return_convergence_delta=True,
        )
        scores = attrs.squeeze(0).sum(-1).abs().detach().cpu().numpy()  # (L,)
        if is_comment:
            comment_attrs = scores
        else:
            code_attrs = scores

    else:
        # Dual encoder: explain comment tokens with code fixed, then code with comment fixed
        from captum.attr import IntegratedGradients

        # ── comment attribution (code fixed) ──────────────────────────────
        c_emb_fn = model.get_comment_word_embeddings if hasattr(model, "get_comment_word_embeddings") else lambda: model.c_enc.embeddings.word_embeddings
        c_emb_layer = c_emb_fn()
        k_emb_layer = model.get_code_word_embeddings() if hasattr(model, "get_code_word_embeddings") else model.k_enc.embeddings.word_embeddings

        c_embeds = c_emb_layer(comment_tk_ids)   # (1, Lc, H)
        k_embeds = k_emb_layer(code_tk_ids)      # (1, Lk, H)

        def fwd_comment_embs(c_emb: torch.Tensor) -> torch.Tensor:
            """Forward with variable comment embeddings, fixed code."""
            # Comment encoder with embeddings
            c_kw: Dict = dict(inputs_embeds=c_emb, attention_mask=comment_attn_mask)
            if comment_ttids is not None:
                c_kw["token_type_ids"] = comment_ttids

            k_kw: Dict = dict(inputs_embeds=k_embeds.detach(), attention_mask=code_attn_mask)
            if code_ttids is not None:
                k_kw["token_type_ids"] = code_ttids

            c_out = model.c_enc(**c_kw)
            k_out = model.k_enc(**k_kw)

            if model_type == "late_fusion":
                c_rep = mean_pool(c_out.last_hidden_state, comment_attn_mask)
                k_rep = mean_pool(k_out.last_hidden_state, code_attn_mask)
                if model.c_proj is not None:
                    c_rep = model.c_proj(c_rep)
                if model.k_proj is not None:
                    k_rep = model.k_proj(k_rep)
                return model.head(torch.cat([c_rep, k_rep], dim=-1))
            else:
                # cross_attn
                c_h = model.c_proj(c_out.last_hidden_state)
                k_h = model.k_proj(k_out.last_hidden_state) if not isinstance(model.k_proj, nn.Identity) else k_out.last_hidden_state
                kpm = (code_attn_mask == 0)
                attn_vals, _ = model.cross_attn(c_h, k_h, k_h, key_padding_mask=kpm)
                c_att = model.norm(c_h + model.drop(attn_vals))
                c_rep = mean_pool(c_att, comment_attn_mask)
                k_rep = mean_pool(k_h, code_attn_mask)
                return model.head(torch.cat([c_rep, k_rep], dim=-1))

        ig_c = IntegratedGradients(fwd_comment_embs)
        c_attrs_raw, _ = ig_c.attribute(
            inputs=c_embeds,
            baselines=torch.zeros_like(c_embeds),
            target=target_class,
            n_steps=n_steps,
            internal_batch_size=internal_batch,
            return_convergence_delta=True,
        )
        comment_attrs = c_attrs_raw.squeeze(0).sum(-1).abs().detach().cpu().numpy()

        # ── code attribution (comment fixed) ──────────────────────────────
        def fwd_code_embs(k_emb: torch.Tensor) -> torch.Tensor:
            c_kw: Dict = dict(inputs_embeds=c_embeds.detach(), attention_mask=comment_attn_mask)
            if comment_ttids is not None:
                c_kw["token_type_ids"] = comment_ttids
            k_kw: Dict = dict(inputs_embeds=k_emb, attention_mask=code_attn_mask)
            if code_ttids is not None:
                k_kw["token_type_ids"] = code_ttids

            c_out = model.c_enc(**c_kw)
            k_out = model.k_enc(**k_kw)

            if model_type == "late_fusion":
                c_rep = mean_pool(c_out.last_hidden_state, comment_attn_mask)
                k_rep = mean_pool(k_out.last_hidden_state, code_attn_mask)
                if model.c_proj is not None:
                    c_rep = model.c_proj(c_rep)
                if model.k_proj is not None:
                    k_rep = model.k_proj(k_rep)
                return model.head(torch.cat([c_rep, k_rep], dim=-1))
            else:
                c_h = model.c_proj(c_out.last_hidden_state)
                k_h = model.k_proj(k_out.last_hidden_state) if not isinstance(model.k_proj, nn.Identity) else k_out.last_hidden_state
                kpm = (code_attn_mask == 0)
                attn_vals, _ = model.cross_attn(c_h, k_h, k_h, key_padding_mask=kpm)
                c_att = model.norm(c_h + model.drop(attn_vals))
                c_rep = mean_pool(c_att, comment_attn_mask)
                k_rep = mean_pool(k_h, code_attn_mask)
                return model.head(torch.cat([c_rep, k_rep], dim=-1))

        ig_k = IntegratedGradients(fwd_code_embs)
        k_attrs_raw, _ = ig_k.attribute(
            inputs=k_embeds,
            baselines=torch.zeros_like(k_embeds),
            target=target_class,
            n_steps=n_steps,
            internal_batch_size=internal_batch,
            return_convergence_delta=True,
        )
        code_attrs = k_attrs_raw.squeeze(0).sum(-1).abs().detach().cpu().numpy()

    return comment_attrs, code_attrs


# ===========================================================================
# Token → line mapping
# ===========================================================================

def tokens_to_lines(
    token_scores: np.ndarray,
    offset_mapping: List[Tuple[int, int]],
    raw_text: str,
    special_ids: set,
    input_ids_list: List[int],
    top_k: int = 10,
    snippet_max: int = 120,
) -> List[Dict]:
    """
    Map top-k code tokens by attribution score to their source lines.

    Returns: list of {line_no (0-based), text (snippet)} deduplicated by line,
             sorted by max per-line attribution descending.
    """
    # Find newline positions
    newline_positions = [i for i, c in enumerate(raw_text) if c == "\n"]

    def char_to_line(char_pos: int) -> int:
        line = 0
        for nl_pos in newline_positions:
            if char_pos > nl_pos:
                line += 1
            else:
                break
        return line

    lines_of_text = raw_text.split("\n")

    # Build token->line map
    token_line_map: List[Optional[int]] = []
    for tok_idx, (start, end) in enumerate(offset_mapping):
        if input_ids_list[tok_idx] in special_ids or (start == 0 and end == 0):
            token_line_map.append(None)
        else:
            token_line_map.append(char_to_line(start))

    # Top-k non-special tokens
    valid_indices = [i for i, ln in enumerate(token_line_map) if ln is not None]
    if not valid_indices:
        return []

    valid_scores = [(i, token_scores[i]) for i in valid_indices]
    valid_scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = [i for i, _ in valid_scores[:top_k]]

    # Aggregate by line
    line_scores: Dict[int, float] = {}
    for i in top_indices:
        ln = token_line_map[i]
        if ln is not None:
            line_scores[ln] = max(line_scores.get(ln, 0.0), float(token_scores[i]))

    result = []
    for ln in sorted(line_scores, key=lambda l: line_scores[l], reverse=True):
        snippet = lines_of_text[ln][:snippet_max] if ln < len(lines_of_text) else ""
        result.append({"line_no": ln, "text": snippet})
    return result


# ===========================================================================
# Faithfulness: deletion & sufficiency
# ===========================================================================

def _get_mask_token_id(tokenizer) -> int:
    if tokenizer.mask_token_id is not None:
        return tokenizer.mask_token_id
    return tokenizer.pad_token_id


def _mask_top_k(input_ids: torch.Tensor, scores: np.ndarray,
                k: int, mask_id: int, special_ids: set) -> torch.Tensor:
    """Return a copy of input_ids with top-k non-special tokens replaced by mask_id."""
    masked = input_ids.clone()
    ids_list = input_ids[0].tolist()
    valid = [(i, scores[i]) for i in range(len(ids_list)) if ids_list[i] not in special_ids]
    valid.sort(key=lambda x: x[1], reverse=True)
    for i, _ in valid[:k]:
        masked[0, i] = mask_id
    return masked


def _keep_only_top_k(input_ids: torch.Tensor, scores: np.ndarray,
                     k: int, mask_id: int, special_ids: set) -> torch.Tensor:
    """Return input_ids with all non-special tokens EXCEPT top-k replaced by mask_id."""
    masked = input_ids.clone()
    ids_list = input_ids[0].tolist()
    valid = [(i, scores[i]) for i in range(len(ids_list)) if ids_list[i] not in special_ids]
    valid.sort(key=lambda x: x[1], reverse=True)
    top_set = {i for i, _ in valid[:k]}
    for i, vid in enumerate(ids_list):
        if vid not in special_ids and i not in top_set:
            masked[0, i] = mask_id
    return masked


def get_prob(model: nn.Module, model_type: str, target: int,
             comment_ids: torch.Tensor, comment_mask: torch.Tensor,
             code_ids: torch.Tensor, code_mask: torch.Tensor,
             comment_ttids=None, code_ttids=None) -> float:
    """Get softmax probability for the target class."""
    model.eval()
    with torch.no_grad():
        if model_type == "comment":
            kw = dict(input_ids=comment_ids, attention_mask=comment_mask)
            if comment_ttids is not None:
                kw["token_type_ids"] = comment_ttids
            logits = model(**kw) if callable(model) else model.forward(**kw)
        elif model_type == "code":
            kw = dict(input_ids=code_ids, attention_mask=code_mask)
            if code_ttids is not None:
                kw["token_type_ids"] = code_ttids
            logits = model(**kw)
        else:
            kw = dict(
                comment_input_ids=comment_ids, comment_attention_mask=comment_mask,
                code_input_ids=code_ids, code_attention_mask=code_mask,
            )
            if comment_ttids is not None:
                kw["comment_token_type_ids"] = comment_ttids
            if code_ttids is not None:
                kw["code_token_type_ids"] = code_ttids
            logits = model(**kw)

        probs = torch.softmax(logits, dim=-1)
        return float(probs[0, target].item())


def compute_faithfulness_metrics(
    model: nn.Module,
    model_type: str,
    target_class: int,
    p_orig: float,
    comment_ids: torch.Tensor,
    comment_mask: torch.Tensor,
    code_ids: torch.Tensor,
    code_mask: torch.Tensor,
    comment_attrs: Optional[np.ndarray],
    code_attrs: Optional[np.ndarray],
    comment_tokenizer,
    code_tokenizer,
    k_values: List[int],
    comment_ttids=None,
    code_ttids=None,
) -> Dict[int, Dict[str, float]]:
    """
    For each k: compute deletion and sufficiency.
    Masking is applied to whichever track has attributions.
    For dual-encoder: mask comment AND code simultaneously.
    """
    c_mask_id = _get_mask_token_id(comment_tokenizer)
    k_mask_id = _get_mask_token_id(code_tokenizer)

    c_special = set(comment_tokenizer.all_special_ids) if hasattr(comment_tokenizer, "all_special_ids") else set()
    k_special = set(code_tokenizer.all_special_ids) if hasattr(code_tokenizer, "all_special_ids") else set()

    results: Dict[int, Dict[str, float]] = {}
    for kk in k_values:
        # ── Deletion (comprehensiveness) ───────────────────────────────────
        del_c_ids = comment_ids
        del_k_ids = code_ids
        if comment_attrs is not None:
            del_c_ids = _mask_top_k(comment_ids, comment_attrs, kk, c_mask_id, c_special)
        if code_attrs is not None:
            del_k_ids = _mask_top_k(code_ids, code_attrs, kk, k_mask_id, k_special)

        p_del = get_prob(model, model_type, target_class,
                         del_c_ids, comment_mask, del_k_ids, code_mask,
                         comment_ttids, code_ttids)
        delta_del = p_orig - p_del

        # ── Sufficiency ────────────────────────────────────────────────────
        suf_c_ids = comment_ids
        suf_k_ids = code_ids
        if comment_attrs is not None:
            suf_c_ids = _keep_only_top_k(comment_ids, comment_attrs, kk, c_mask_id, c_special)
        if code_attrs is not None:
            suf_k_ids = _keep_only_top_k(code_ids, code_attrs, kk, k_mask_id, k_special)

        p_suf = get_prob(model, model_type, target_class,
                         suf_c_ids, comment_mask, suf_k_ids, code_mask,
                         comment_ttids, code_ttids)
        delta_suf = p_orig - p_suf

        results[kk] = {
            "delta_del": float(delta_del),
            "p_suf":     float(p_suf),
            "delta_suf": float(delta_suf),
        }
    return results


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    set_seed(42)
    device = torch.device(CONFIG["DEVICE"])
    output_dir = Path(CONFIG["OUTPUT_DIR"])
    best_dir   = Path(CONFIG["BEST_MODEL_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load best config ─────────────────────────────────────────────────
    best_cfg_path = best_dir / "best_config.json"
    if not best_cfg_path.exists():
        raise FileNotFoundError(f"best_config.json not found at {best_cfg_path}")
    with open(best_cfg_path) as f:
        best_cfg = json.load(f)
    logger.info("Loaded best_config: type=%s  comment=%s  code=%s",
                best_cfg["model_type"], best_cfg.get("comment_encoder"),
                best_cfg.get("code_encoder"))

    label2id: Dict[str, int] = best_cfg["label2id"]
    id2label: Dict[int, str] = {int(k): v for k, v in best_cfg["id2label"].items()}
    labels: List[str] = best_cfg["labels"]
    num_labels: int = best_cfg["num_labels"]

    # ── 2. Build & load model ───────────────────────────────────────────────
    model = build_model(best_cfg)
    ckpt_path = best_dir / "model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    logger.info("Model loaded from %s", ckpt_path)

    model_type = best_cfg["model_type"]

    # ── 3. Tokenisers ───────────────────────────────────────────────────────
    c_enc_name = best_cfg.get("comment_encoder") or best_cfg.get("code_encoder")
    k_enc_name = best_cfg.get("code_encoder") or best_cfg.get("comment_encoder")
    comment_tokenizer = get_tokenizer(c_enc_name)
    code_tokenizer    = get_tokenizer(k_enc_name)

    # ── 4. Rebuild test split ───────────────────────────────────────────────
    split_info_path = best_cfg.get("split_info_path", str(output_dir / "split_info.json"))
    if not Path(split_info_path).exists():
        split_info_path = str(output_dir / "split_info.json")

    df_raw = load_dataframe(CONFIG["DATA_PATH"])
    label_aliases = CONFIG["LABEL_ALIASES"]

    rows = []
    for _, row in df_raw.iterrows():
        s = normalize_sample(row, label_aliases, label2id)
        if s:
            rows.append(s)
    df = pd.DataFrame(rows)

    if Path(split_info_path).exists():
        with open(split_info_path) as f:
            split_info = json.load(f)
        test_ids = set(split_info["test_comment_ids"])
        test_df = df[df["comment_id"].isin(test_ids)].reset_index(drop=True)
        logger.info("Restored test split: %d samples", len(test_df))
    else:
        logger.warning("split_info.json not found — using last 10%% as test.")
        n_test = max(1, int(0.10 * len(df)))
        test_df = df.tail(n_test).reset_index(drop=True)

    # ── 5. Stratified subset for XAI ───────────────────────────────────────
    xai_df = stratified_subset(
        test_df, CONFIG["XAI_SUBSET_SIZE"], label_col="label_id",
        num_labels=num_labels, seed=best_cfg.get("seed", 42)
    )
    logger.info("XAI subset: %d samples", len(xai_df))

    # ── 6. Run IG + faithfulness ────────────────────────────────────────────
    k_values = CONFIG["FAITHFULNESS_K"]
    top_report = CONFIG["TOP_TOKENS_REPORT"]
    max_c = best_cfg.get("max_len_comment", 128)
    max_k = best_cfg.get("max_len_code", 384)

    per_sample: List[Dict] = []
    # Accumulators for summary
    k_agg: Dict[int, Dict[str, List[float]]] = {
        kk: {"delta_del": [], "p_suf": [], "delta_suf": []} for kk in k_values
    }
    per_class_k_agg: Dict[str, Dict[int, Dict[str, List[float]]]] = defaultdict(
        lambda: {kk: {"delta_del": [], "p_suf": [], "delta_suf": []} for kk in k_values}
    )

    for idx, sample_row in xai_df.iterrows():
        comment_text = sample_row["comment"]
        code_text    = sample_row["code"]
        true_label   = sample_row["label"]
        comment_id   = sample_row["comment_id"]

        # Tokenise
        c_enc_out = tokenize_single(comment_tokenizer, comment_text, max_c, device)
        k_enc_out = tokenize_single(code_tokenizer, code_text, max_k, device)

        c_ids   = c_enc_out["input_ids"]           # (1, Lc)
        c_amask = c_enc_out["attention_mask"]
        c_ttids = c_enc_out.get("token_type_ids")
        c_offsets = c_enc_out["offset_mapping"][0].cpu().tolist()

        k_ids   = k_enc_out["input_ids"]
        k_amask = k_enc_out["attention_mask"]
        k_ttids = k_enc_out.get("token_type_ids")
        k_offsets = k_enc_out["offset_mapping"][0].cpu().tolist()

        # Predict
        with torch.no_grad():
            if model_type == "comment":
                kw = dict(input_ids=c_ids, attention_mask=c_amask)
                if c_ttids is not None:
                    kw["token_type_ids"] = c_ttids
                logits = model(**kw)
            elif model_type == "code":
                kw = dict(input_ids=k_ids, attention_mask=k_amask)
                if k_ttids is not None:
                    kw["token_type_ids"] = k_ttids
                logits = model(**kw)
            else:
                kw = dict(
                    comment_input_ids=c_ids, comment_attention_mask=c_amask,
                    code_input_ids=k_ids, code_attention_mask=k_amask,
                )
                if c_ttids is not None:
                    kw["comment_token_type_ids"] = c_ttids
                if k_ttids is not None:
                    kw["code_token_type_ids"] = k_ttids
                logits = model(**kw)

        probs = torch.softmax(logits, dim=-1)
        pred_class = int(torch.argmax(probs, dim=-1).item())
        pred_prob  = float(probs[0, pred_class].item())
        p_orig     = float(probs[0, pred_class].item())
        pred_label = id2label[pred_class]

        # IG attributions
        try:
            c_attrs, k_attrs = compute_ig_attributions(
                model=model,
                model_type=model_type,
                comment_enc=model.encoder if hasattr(model, "encoder") else None,
                code_enc=model.encoder if model_type == "code" else None,
                comment_tk_ids=c_ids,
                comment_attn_mask=c_amask,
                code_tk_ids=k_ids,
                code_attn_mask=k_amask,
                target_class=pred_class,
                n_steps=CONFIG["IG_N_STEPS"],
                internal_batch=CONFIG["IG_INTERNAL_BATCH"],
                comment_ttids=c_ttids,
                code_ttids=k_ttids,
            )
        except Exception as e:
            logger.warning("IG failed for sample %s: %s", comment_id, e)
            c_attrs, k_attrs = None, None

        # Top comment tokens
        c_special = set(comment_tokenizer.all_special_ids) if hasattr(comment_tokenizer, "all_special_ids") else set()
        k_special = set(code_tokenizer.all_special_ids) if hasattr(code_tokenizer, "all_special_ids") else set()

        top_comment_tokens: List[str] = []
        if c_attrs is not None:
            c_ids_list = c_ids[0].tolist()
            valid_c = [(i, c_attrs[i]) for i in range(len(c_ids_list)) if c_ids_list[i] not in c_special]
            valid_c.sort(key=lambda x: x[1], reverse=True)
            top_comment_tokens = [
                comment_tokenizer.convert_ids_to_tokens(c_ids_list[i]) or ""
                for i, _ in valid_c[:top_report]
            ]

        # Top code lines
        top_code_lines: List[Dict] = []
        if k_attrs is not None:
            k_ids_list = k_ids[0].tolist()
            top_code_lines = tokens_to_lines(
                k_attrs, k_offsets, code_text, k_special, k_ids_list,
                top_k=top_report, snippet_max=120,
            )

        # Faithfulness metrics
        try:
            faith_metrics = compute_faithfulness_metrics(
                model=model,
                model_type=model_type,
                target_class=pred_class,
                p_orig=p_orig,
                comment_ids=c_ids,
                comment_mask=c_amask,
                code_ids=k_ids,
                code_mask=k_amask,
                comment_attrs=c_attrs,
                code_attrs=k_attrs,
                comment_tokenizer=comment_tokenizer,
                code_tokenizer=code_tokenizer,
                k_values=k_values,
                comment_ttids=c_ttids,
                code_ttids=k_ttids,
            )
        except Exception as e:
            logger.warning("Faithfulness failed for sample %s: %s", comment_id, e)
            faith_metrics = {kk: {"delta_del": 0.0, "p_suf": 0.0, "delta_suf": 0.0} for kk in k_values}

        # Accumulate
        for kk, metrics in faith_metrics.items():
            for metric_name, val in metrics.items():
                k_agg[kk][metric_name].append(val)
                per_class_k_agg[true_label][kk][metric_name].append(val)

        per_sample.append({
            "comment_id":         comment_id,
            "true_label":         true_label,
            "pred_label":         pred_label,
            "pred_prob":          pred_prob,
            "top_comment_tokens": top_comment_tokens,
            "top_code_lines":     top_code_lines,
            "k_metrics":          {str(kk): v for kk, v in faith_metrics.items()},
        })

        if (idx + 1) % 50 == 0:
            logger.info("Processed %d / %d samples", idx + 1, len(xai_df))

    # ── 7. Aggregate summary ────────────────────────────────────────────────
    def _agg(lst: List[float]) -> float:
        return float(np.mean(lst)) if lst else 0.0

    summary: Dict[str, Any] = {"overall": {}, "per_class": {}}
    for kk in k_values:
        summary["overall"][str(kk)] = {
            "mean_delta_del": _agg(k_agg[kk]["delta_del"]),
            "mean_p_suf":     _agg(k_agg[kk]["p_suf"]),
            "mean_delta_suf": _agg(k_agg[kk]["delta_suf"]),
            "n_samples":      len(k_agg[kk]["delta_del"]),
        }

    for cls_name, cls_data in per_class_k_agg.items():
        summary["per_class"][cls_name] = {}
        for kk in k_values:
            summary["per_class"][cls_name][str(kk)] = {
                "mean_delta_del": _agg(cls_data[kk]["delta_del"]),
                "mean_p_suf":     _agg(cls_data[kk]["p_suf"]),
                "mean_delta_suf": _agg(cls_data[kk]["delta_suf"]),
                "n_samples":      len(cls_data[kk]["delta_del"]),
            }

    # ── 8. Save outputs ─────────────────────────────────────────────────────
    faith_path = output_dir / "faithfulness_summary.json"
    with open(faith_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved faithfulness_summary.json")

    expl_path = output_dir / "per_sample_explanations.jsonl"
    with open(expl_path, "w") as f:
        for s in per_sample:
            f.write(json.dumps(s) + "\n")
    logger.info("Saved per_sample_explanations.jsonl (%d lines)", len(per_sample))

    logger.info("\n==== Faithfulness Summary (overall) ====")
    for kk in k_values:
        ov = summary["overall"][str(kk)]
        logger.info(
            "  k=%2d | mean_delta_del=%.4f | mean_p_suf=%.4f | mean_delta_suf=%.4f | n=%d",
            kk, ov["mean_delta_del"], ov["mean_p_suf"], ov["mean_delta_suf"], ov["n_samples"]
        )

    logger.info("Done. Outputs at: %s", output_dir)


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    main()
