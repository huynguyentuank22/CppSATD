"""
models_fusion.py
----------------
Dual-encoder fusion models for SATD detection.
  - DualEncoderLateFusion         : concat pooled reps + classifier
  - DualEncoderCrossAttentionFusion: comment attends over code via MHA

All configuration via top-level CONFIG dict.
No argparse; no dependency on other project files.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModel

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CONFIG: Dict = {
    "LABELS": ["Non-SATD", "Design", "Requirement", "Defect", "Test", "Documentation"],
    "LABEL2ID": {},
    "ID2LABEL": {},
    "COMMENT_ENCODERS": [
        "roberta-base",
        "microsoft/deberta-v3-base",
        "bert-base-uncased",
    ],
    "CODE_ENCODERS": [
        "microsoft/codebert-base",
        "microsoft/graphcodebert-base",
        "microsoft/unixcoder-base-nine",
    ],
    # Fusion defaults
    "DROPOUT": 0.1,
    "POOLING": "mean",
    "CROSS_ATTN_HEADS": 8,
    # Projection dim for cross-attention when hidden dims differ
    "PROJ_DIM": 768,
}

CONFIG["LABEL2ID"] = {lbl: i for i, lbl in enumerate(CONFIG["LABELS"])}
CONFIG["ID2LABEL"] = {i: lbl for lbl, i in CONFIG["LABEL2ID"].items()}
NUM_LABELS: int = len(CONFIG["LABELS"])

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Shared utilities (duplicated here so file is self-contained)
# ---------------------------------------------------------------------------

def _mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def _cls_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:  # noqa: ARG001
    return token_embeddings[:, 0, :]


def _pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor, strategy: str = "mean") -> torch.Tensor:
    return _mean_pool(token_embeddings, attention_mask) if strategy == "mean" else _cls_pool(token_embeddings, attention_mask)


class _ClassifierHead(nn.Module):
    def __init__(self, in_dim: int, num_labels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_dim, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.drop(x))


# ---------------------------------------------------------------------------
# 1. Late Fusion
# ---------------------------------------------------------------------------

class DualEncoderLateFusion(nn.Module):
    """
    Independently encode comment and code, concatenate pooled representations,
    optionally project to a common dim, then classify.

    If hidden sizes differ the smaller representation is projected to match
    the larger (or to PROJ_DIM if both differ from it) so the concat dimension
    is always 2 * fusion_dim.
    """

    def __init__(
        self,
        comment_model_name: str,
        code_model_name: str,
        num_labels: int = NUM_LABELS,
        dropout: float = CONFIG["DROPOUT"],
        pooling: str = CONFIG["POOLING"],
        class_weights: Optional[torch.Tensor] = None,
        proj_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.comment_model_name = comment_model_name
        self.code_model_name = code_model_name
        self.num_labels = num_labels
        self.pooling = pooling
        self.class_weights = class_weights

        logger.info("LateFusion | comment=%s  code=%s", comment_model_name, code_model_name)
        self.comment_encoder = AutoModel.from_pretrained(comment_model_name)
        self.code_encoder = AutoModel.from_pretrained(code_model_name)

        c_dim: int = self.comment_encoder.config.hidden_size
        k_dim: int = self.code_encoder.config.hidden_size

        # Optionally project to a common dim
        if proj_dim is not None:
            self.comment_proj: Optional[nn.Linear] = nn.Linear(c_dim, proj_dim)
            self.code_proj: Optional[nn.Linear] = nn.Linear(k_dim, proj_dim)
            fusion_dim = proj_dim
        elif c_dim != k_dim:
            # project to the larger dim
            target = max(c_dim, k_dim)
            self.comment_proj = nn.Linear(c_dim, target) if c_dim != target else None
            self.code_proj = nn.Linear(k_dim, target) if k_dim != target else None
            fusion_dim = target
        else:
            self.comment_proj = None
            self.code_proj = None
            fusion_dim = c_dim

        self.head = _ClassifierHead(2 * fusion_dim, num_labels, dropout)
        self._fusion_dim = fusion_dim

    # ------------------------------------------------------------------
    def _encode_comment(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kwargs: Dict = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.comment_encoder(**kwargs)
        rep = _pool(out.last_hidden_state, attention_mask, self.pooling)
        if self.comment_proj is not None:
            rep = self.comment_proj(rep)
        return rep

    def _encode_code(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kwargs: Dict = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.code_encoder(**kwargs)
        rep = _pool(out.last_hidden_state, attention_mask, self.pooling)
        if self.code_proj is not None:
            rep = self.code_proj(rep)
        return rep

    # ------------------------------------------------------------------
    def forward(
        self,
        comment_input_ids: torch.Tensor,
        comment_attention_mask: torch.Tensor,
        code_input_ids: torch.Tensor,
        code_attention_mask: torch.Tensor,
        comment_token_type_ids: Optional[torch.Tensor] = None,
        code_token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Returns dict with 'logits' and optionally 'loss'."""
        c_rep = self._encode_comment(comment_input_ids, comment_attention_mask, comment_token_type_ids)
        k_rep = self._encode_code(code_input_ids, code_attention_mask, code_token_type_ids)

        fused = torch.cat([c_rep, k_rep], dim=-1)   # (B, 2*fusion_dim)
        logits = self.head(fused)

        result: Dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            weights = self.class_weights.to(logits.device) if self.class_weights is not None else None
            result["loss"] = nn.CrossEntropyLoss(weight=weights)(logits, labels)
        return result

    def get_comment_embeddings(self) -> nn.Embedding:
        try:
            return self.comment_encoder.embeddings.word_embeddings
        except AttributeError:
            return self.comment_encoder.embeddings.word_embeddings

    def get_code_embeddings(self) -> nn.Embedding:
        try:
            return self.code_encoder.embeddings.word_embeddings
        except AttributeError:
            return self.code_encoder.embeddings.word_embeddings


# ---------------------------------------------------------------------------
# 2. Cross-Attention Fusion
# ---------------------------------------------------------------------------

class DualEncoderCrossAttentionFusion(nn.Module):
    """
    Encode comment and code independently.
    Use Multi-Head Cross-Attention: comment tokens as Q, code tokens as K/V.
    Pool the attended comment context + code CLS, concatenate, classify.

    Hidden-size mismatch: both sides projected to proj_dim before attention.
    """

    def __init__(
        self,
        comment_model_name: str,
        code_model_name: str,
        num_labels: int = NUM_LABELS,
        heads: int = CONFIG["CROSS_ATTN_HEADS"],
        dropout: float = CONFIG["DROPOUT"],
        pooling: str = CONFIG["POOLING"],
        proj_dim: int = CONFIG["PROJ_DIM"],
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.comment_model_name = comment_model_name
        self.code_model_name = code_model_name
        self.num_labels = num_labels
        self.pooling = pooling
        self.class_weights = class_weights
        self.proj_dim = proj_dim

        logger.info("CrossAttnFusion | comment=%s  code=%s  heads=%d  proj_dim=%d",
                    comment_model_name, code_model_name, heads, proj_dim)

        self.comment_encoder = AutoModel.from_pretrained(comment_model_name)
        self.code_encoder = AutoModel.from_pretrained(code_model_name)

        c_dim: int = self.comment_encoder.config.hidden_size
        k_dim: int = self.code_encoder.config.hidden_size

        # Projection layers (always project to proj_dim for safe head divisibility)
        self.comment_proj = nn.Linear(c_dim, proj_dim) if c_dim != proj_dim else nn.Identity()
        self.code_proj = nn.Linear(k_dim, proj_dim) if k_dim != proj_dim else nn.Identity()

        # Ensure proj_dim divisible by heads; if not, round down to nearest divisible value
        safe_heads = heads
        while proj_dim % safe_heads != 0 and safe_heads > 1:
            safe_heads -= 1
        if safe_heads != heads:
            logger.warning("Adjusted cross-attn heads from %d to %d for proj_dim=%d", heads, safe_heads, proj_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=safe_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(proj_dim)
        self.dropout_layer = nn.Dropout(dropout)

        # Classifier on [cross-attended comment rep; code rep] => 2 * proj_dim
        self.head = _ClassifierHead(2 * proj_dim, num_labels, dropout)

    # ------------------------------------------------------------------
    def forward(
        self,
        comment_input_ids: torch.Tensor,
        comment_attention_mask: torch.Tensor,
        code_input_ids: torch.Tensor,
        code_attention_mask: torch.Tensor,
        comment_token_type_ids: Optional[torch.Tensor] = None,
        code_token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Encode comment
        c_kwargs: Dict = dict(input_ids=comment_input_ids, attention_mask=comment_attention_mask)
        if comment_token_type_ids is not None:
            c_kwargs["token_type_ids"] = comment_token_type_ids
        c_out = self.comment_encoder(**c_kwargs)
        c_hidden = self.comment_proj(c_out.last_hidden_state)    # (B, Lc, P)

        # Encode code
        k_kwargs: Dict = dict(input_ids=code_input_ids, attention_mask=code_attention_mask)
        if code_token_type_ids is not None:
            k_kwargs["token_type_ids"] = code_token_type_ids
        k_out = self.code_encoder(**k_kwargs)
        k_hidden = self.code_proj(k_out.last_hidden_state)        # (B, Lk, P)

        # Build key_padding_mask for cross-attn (True = ignored position)
        # nn.MultiheadAttention key_padding_mask: (B, Lk), True means ignore
        code_key_padding_mask = (code_attention_mask == 0)         # (B, Lk)

        # Cross-attention: Q=comment, K/V=code
        # attn_output: (B, Lc, P)
        attn_output, _ = self.cross_attn(
            query=c_hidden,
            key=k_hidden,
            value=k_hidden,
            key_padding_mask=code_key_padding_mask,
        )

        # Residual + norm on comment side
        c_attended = self.norm(c_hidden + self.dropout_layer(attn_output))  # (B, Lc, P)

        # Pool attended comment representation
        c_rep = _pool(c_attended, comment_attention_mask, self.pooling)     # (B, P)

        # Pool code representation (unprojected? no — use projected k_hidden for consistency)
        k_rep = _pool(k_hidden, code_attention_mask, self.pooling)          # (B, P)

        fused = torch.cat([c_rep, k_rep], dim=-1)                           # (B, 2P)
        logits = self.head(fused)

        result: Dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            weights = self.class_weights.to(logits.device) if self.class_weights is not None else None
            result["loss"] = nn.CrossEntropyLoss(weight=weights)(logits, labels)
        return result

    def get_comment_embeddings(self) -> nn.Embedding:
        try:
            return self.comment_encoder.embeddings.word_embeddings
        except AttributeError:
            return self.comment_encoder.embeddings.word_embeddings

    def get_code_embeddings(self) -> nn.Embedding:
        try:
            return self.code_encoder.embeddings.word_embeddings
        except AttributeError:
            return self.code_encoder.embeddings.word_embeddings


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("CONFIG labels: %s", CONFIG["LABELS"])
    logger.info("models_fusion.py loaded successfully.")
