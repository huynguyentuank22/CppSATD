"""
models_baselines.py
-------------------
Comment-only and Code-only baseline classifiers for SATD detection.
All configuration is set via top-level CONFIG dict.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CONFIG: Dict = {
    # Label schema (6-class)
    "LABELS": ["Non-SATD", "Design", "Requirement", "Defect", "Test", "Documentation"],
    # Mapping helpers (built below)
    "LABEL2ID": {},
    "ID2LABEL": {},
    # Encoders available for comment track
    "COMMENT_ENCODERS": [
        "roberta-base",
        "microsoft/deberta-v3-base",
        "bert-base-uncased",
    ],
    # Encoders available for code track
    "CODE_ENCODERS": [
        "microsoft/codebert-base",
        "microsoft/graphcodebert-base",
        "microsoft/unixcoder-base-nine",
    ],
    # Default dropout for classifier head
    "DROPOUT": 0.1,
    # Pooling strategy: "mean" or "cls"
    "POOLING": "mean",
}

# Build label maps
CONFIG["LABEL2ID"] = {lbl: i for i, lbl in enumerate(CONFIG["LABELS"])}
CONFIG["ID2LABEL"] = {i: lbl for lbl, i in CONFIG["LABEL2ID"].items()}
NUM_LABELS: int = len(CONFIG["LABELS"])

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Pooling helpers
# ---------------------------------------------------------------------------

def mean_pool(
    token_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Mean-pool token embeddings, ignoring padding tokens.

    Args:
        token_embeddings: (batch, seq_len, hidden)
        attention_mask:   (batch, seq_len)

    Returns:
        pooled: (batch, hidden)
    """
    mask_expanded = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
    sum_emb = (token_embeddings * mask_expanded).sum(dim=1)  # (B, H)
    count = mask_expanded.sum(dim=1).clamp(min=1e-9)          # (B, 1)
    return sum_emb / count


def cls_pool(
    token_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,  # noqa: ARG001  (kept for API symmetry)
) -> torch.Tensor:
    """Return [CLS] token embedding (first position)."""
    return token_embeddings[:, 0, :]


def pool(
    token_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    strategy: str = "mean",
) -> torch.Tensor:
    if strategy == "mean":
        return mean_pool(token_embeddings, attention_mask)
    elif strategy == "cls":
        return cls_pool(token_embeddings, attention_mask)
    else:
        raise ValueError(f"Unknown pooling strategy: {strategy!r}")


# ---------------------------------------------------------------------------
# Classifier head
# ---------------------------------------------------------------------------

class ClassifierHead(nn.Module):
    """Simple dropout + linear classification head."""

    def __init__(self, hidden_size: int, num_labels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.dropout(x))


# ---------------------------------------------------------------------------
# Comment-only classifier
# ---------------------------------------------------------------------------

class CommentOnlyClassifier(nn.Module):
    """
    Encode a SATD comment with a pre-trained LM, pool, then classify.

    Args:
        comment_encoder_name: HuggingFace model ID for the comment encoder.
        num_labels: number of output classes.
        dropout: dropout before classifier.
        pooling: "mean" or "cls".
        class_weights: optional 1-D tensor for weighted cross-entropy.
    """

    def __init__(
        self,
        comment_encoder_name: str,
        num_labels: int = NUM_LABELS,
        dropout: float = CONFIG["DROPOUT"],
        pooling: str = CONFIG["POOLING"],
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.comment_encoder_name = comment_encoder_name
        self.num_labels = num_labels
        self.pooling = pooling
        self.class_weights = class_weights

        logger.info("Loading comment encoder: %s", comment_encoder_name)
        self.encoder = AutoModel.from_pretrained(comment_encoder_name)
        hidden_size: int = self.encoder.config.hidden_size

        self.head = ClassifierHead(hidden_size, num_labels, dropout)

    # ------------------------------------------------------------------
    def forward(
        self,
        comment_input_ids: torch.Tensor,
        comment_attention_mask: torch.Tensor,
        comment_token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with keys "logits" and optionally "loss".
        """
        enc_kwargs: Dict = dict(
            input_ids=comment_input_ids,
            attention_mask=comment_attention_mask,
        )
        if comment_token_type_ids is not None:
            enc_kwargs["token_type_ids"] = comment_token_type_ids

        outputs = self.encoder(**enc_kwargs)
        # last_hidden_state: (B, L, H)
        pooled = pool(outputs.last_hidden_state, comment_attention_mask, self.pooling)
        logits = self.head(pooled)  # (B, num_labels)

        result: Dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            weights = (
                self.class_weights.to(logits.device)
                if self.class_weights is not None
                else None
            )
            loss_fn = nn.CrossEntropyLoss(weight=weights)
            result["loss"] = loss_fn(logits, labels)
        return result

    # ------------------------------------------------------------------
    def get_input_embeddings(self) -> nn.Embedding:
        """Return the word-embedding layer for Captum IG hooks."""
        # Works for BERT/RoBERTa/DeBERTa family
        try:
            return self.encoder.embeddings.word_embeddings
        except AttributeError:
            # UniXcoder / CodeBERT style
            return self.encoder.embeddings.word_embeddings


# ---------------------------------------------------------------------------
# Code-only classifier
# ---------------------------------------------------------------------------

class CodeOnlyClassifier(nn.Module):
    """
    Encode a code-context snippet with a pre-trained code LM.

    Args:
        code_encoder_name: HuggingFace model ID for the code encoder.
        num_labels: number of output classes.
        dropout: dropout before classifier.
        pooling: "mean" or "cls".
        class_weights: optional 1-D tensor for weighted cross-entropy.
    """

    def __init__(
        self,
        code_encoder_name: str,
        num_labels: int = NUM_LABELS,
        dropout: float = CONFIG["DROPOUT"],
        pooling: str = CONFIG["POOLING"],
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.code_encoder_name = code_encoder_name
        self.num_labels = num_labels
        self.pooling = pooling
        self.class_weights = class_weights

        logger.info("Loading code encoder: %s", code_encoder_name)
        self.encoder = AutoModel.from_pretrained(code_encoder_name)
        hidden_size: int = self.encoder.config.hidden_size

        self.head = ClassifierHead(hidden_size, num_labels, dropout)

    # ------------------------------------------------------------------
    def forward(
        self,
        code_input_ids: torch.Tensor,
        code_attention_mask: torch.Tensor,
        code_token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        enc_kwargs: Dict = dict(
            input_ids=code_input_ids,
            attention_mask=code_attention_mask,
        )
        if code_token_type_ids is not None:
            enc_kwargs["token_type_ids"] = code_token_type_ids

        outputs = self.encoder(**enc_kwargs)
        pooled = pool(outputs.last_hidden_state, code_attention_mask, self.pooling)
        logits = self.head(pooled)

        result: Dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            weights = (
                self.class_weights.to(logits.device)
                if self.class_weights is not None
                else None
            )
            loss_fn = nn.CrossEntropyLoss(weight=weights)
            result["loss"] = loss_fn(logits, labels)
        return result

    # ------------------------------------------------------------------
    def get_input_embeddings(self) -> nn.Embedding:
        try:
            return self.encoder.embeddings.word_embeddings
        except AttributeError:
            return self.encoder.embeddings.word_embeddings


# ---------------------------------------------------------------------------
# Quick sanity check (run with: python models_baselines.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("CONFIG labels: %s", CONFIG["LABELS"])
    logger.info("LABEL2ID: %s", CONFIG["LABEL2ID"])
    logger.info("Comment encoders: %s", CONFIG["COMMENT_ENCODERS"])
    logger.info("Code encoders:    %s", CONFIG["CODE_ENCODERS"])
    logger.info("models_baselines.py loaded successfully.")
