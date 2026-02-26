"""
train_eval.py
-------------
Training and evaluation utilities for SATD detection.
Provides:
  - set_seed()
  - compute_class_weights()
  - compute_metrics()
  - train_one_epoch()
  - evaluate()
  - run_training()   — full training loop with early stopping

All configuration via top-level CONFIG dict.
No argparse; self-contained (no imports from other project files).
"""

from __future__ import annotations

import logging
import math
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
)
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CONFIG: Dict = {
    # Reproducibility
    "SEED": 42,
    # Device — auto-detected below
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    # Mixed precision (only meaningful on CUDA)
    "USE_AMP": torch.cuda.is_available(),
    # Training defaults
    "EPOCHS": 3,
    "BATCH_SIZE": 8,
    "LR": 2e-5,
    "WARMUP_RATIO": 0.06,
    "WEIGHT_DECAY": 0.01,
    # Early stopping
    "ES_PATIENCE": 2,
    # Labels
    "LABELS": ["Non-SATD", "Design", "Requirement", "Defect", "Test", "Documentation"],
    "NUM_LABELS": 6,
    # Gradient clipping
    "MAX_GRAD_NORM": 1.0,
}

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed: int = CONFIG["SEED"]) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Seed fixed to %d", seed)


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

def compute_class_weights(
    labels: List[int],
    num_labels: int = CONFIG["NUM_LABELS"],
) -> torch.Tensor:
    """
    Inverse-frequency class weights.

    Args:
        labels: integer label list (train split).
        num_labels: total number of classes.

    Returns:
        1-D float tensor of shape (num_labels,).
    """
    counts = np.bincount(labels, minlength=num_labels).astype(float)
    counts = np.maximum(counts, 1.0)            # avoid div-by-zero for missing classes
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_labels  # scale so mean weight == 1
    logger.info("Class counts: %s", counts.tolist())
    logger.info("Class weights: %s", np.round(weights, 4).tolist())
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    num_labels: int = CONFIG["NUM_LABELS"],
    label_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute macro-F1, per-class F1, and confusion matrix.

    Returns:
        dict with keys: macro_f1, per_class_f1 (dict label->f1), confusion_matrix (list[list]).
    """
    labels_used = list(range(num_labels))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels_used, average="macro", zero_division=0))
    per_class = f1_score(y_true, y_pred, labels=labels_used, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels_used).tolist()

    names = label_names if label_names else [str(i) for i in labels_used]
    per_class_dict = {names[i]: float(per_class[i]) for i in range(len(names))}

    report = classification_report(y_true, y_pred, labels=labels_used,
                                   target_names=names, zero_division=0)
    logger.info("\n%s", report)

    return {
        "macro_f1": macro_f1,
        "per_class_f1": per_class_dict,
        "confusion_matrix": cm,
    }


# ---------------------------------------------------------------------------
# Single epoch training
# ---------------------------------------------------------------------------

def _batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    class_weights: Optional[torch.Tensor] = None,
    max_grad_norm: float = CONFIG["MAX_GRAD_NORM"],
) -> float:
    """
    Run one pass over the training DataLoader.

    Args:
        model: callable that accepts **batch_dict and returns dict with 'loss'.
        loader: training DataLoader.
        optimizer, scheduler: standard torch objects.
        device: target device.
        scaler: GradScaler for AMP (None = disabled).
        class_weights: moved to device inside model; passed here only for
                       models that accept it as forward() kwarg (optional).
        max_grad_norm: gradient clipping threshold.

    Returns:
        mean training loss over epoch.
    """
    model.train()
    total_loss = 0.0
    steps = 0

    for batch in loader:
        batch = _batch_to_device(batch, device)
        optimizer.zero_grad()

        use_amp = scaler is not None
        with autocast(enabled=use_amp):
            output = model(**batch)
            loss = output["loss"]

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()
        steps += 1

    mean_loss = total_loss / max(steps, 1)
    logger.info("  train_loss=%.4f  steps=%d", mean_loss, steps)
    return mean_loss


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run model in eval mode over loader.

    Returns:
        logits (N, num_labels), preds (N,), labels (N,)  — all numpy arrays.
    """
    model.eval()
    all_logits: List[np.ndarray] = []
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for batch in loader:
            labels_batch = batch.get("labels")
            batch = _batch_to_device(batch, device)
            output = model(**batch)
            logits = output["logits"]  # (B, C)
            preds = torch.argmax(logits, dim=-1)

            all_logits.append(logits.cpu().numpy())
            all_preds.extend(preds.cpu().tolist())
            if labels_batch is not None:
                all_labels.extend(labels_batch.tolist())

    logits_arr = np.concatenate(all_logits, axis=0)
    preds_arr = np.array(all_preds)
    labels_arr = np.array(all_labels) if all_labels else np.zeros(len(preds_arr), dtype=int)
    return logits_arr, preds_arr, labels_arr


# ---------------------------------------------------------------------------
# Full training loop with early stopping
# ---------------------------------------------------------------------------

def build_optimizer_scheduler(
    model: nn.Module,
    num_training_steps: int,
    lr: float = CONFIG["LR"],
    warmup_ratio: float = CONFIG["WARMUP_RATIO"],
    weight_decay: float = CONFIG["WEIGHT_DECAY"],
) -> Tuple[torch.optim.Optimizer, Any]:
    """Build AdamW optimizer and linear warmup scheduler."""
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    grouped_params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(grouped_params, lr=lr)
    warmup_steps = max(1, int(num_training_steps * warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )
    return optimizer, scheduler


def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = CONFIG["EPOCHS"],
    lr: float = CONFIG["LR"],
    warmup_ratio: float = CONFIG["WARMUP_RATIO"],
    weight_decay: float = CONFIG["WEIGHT_DECAY"],
    es_patience: int = CONFIG["ES_PATIENCE"],
    use_amp: bool = CONFIG["USE_AMP"],
    num_labels: int = CONFIG["NUM_LABELS"],
    label_names: Optional[List[str]] = None,
    max_grad_norm: float = CONFIG["MAX_GRAD_NORM"],
) -> Dict[str, Any]:
    """
    Full training loop with early stopping on val macro-F1.

    Returns:
        dict with keys:
            best_val_macro_f1, best_val_metrics, best_epoch,
            best_state_dict (on CPU), history (list per epoch).
    """
    model.to(device)
    num_training_steps = epochs * len(train_loader)
    optimizer, scheduler = build_optimizer_scheduler(
        model, num_training_steps, lr, warmup_ratio, weight_decay
    )
    scaler: Optional[GradScaler] = GradScaler() if use_amp else None

    best_val_f1 = -1.0
    best_state: Optional[Dict] = None
    best_metrics: Optional[Dict] = None
    best_epoch = 0
    patience_ctr = 0
    history: List[Dict] = []

    for epoch in range(1, epochs + 1):
        logger.info("=== Epoch %d / %d ===", epoch, epochs)
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, scaler, max_grad_norm=max_grad_norm
        )

        _, val_preds, val_labels = evaluate(model, val_loader, device)
        val_metrics = compute_metrics(val_labels.tolist(), val_preds.tolist(), num_labels, label_names)
        val_f1 = val_metrics["macro_f1"]
        logger.info("  val_macro_f1=%.4f  (best=%.4f)", val_f1, best_val_f1)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_macro_f1": val_f1})

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_metrics = val_metrics
            best_epoch = epoch
            # Save state dict to CPU to avoid OOM when switching models
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
            logger.info("  --> New best model saved (epoch %d, macro_f1=%.4f)", epoch, val_f1)
        else:
            patience_ctr += 1
            logger.info("  No improvement. patience=%d/%d", patience_ctr, es_patience)
            if patience_ctr >= es_patience:
                logger.info("Early stopping triggered at epoch %d.", epoch)
                break

    return {
        "best_val_macro_f1": best_val_f1,
        "best_val_metrics": best_metrics,
        "best_epoch": best_epoch,
        "best_state_dict": best_state,
        "history": history,
    }


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    set_seed(42)
    weights = compute_class_weights([0, 0, 1, 2, 3, 4, 5, 0], num_labels=6)
    logger.info("Sample class weights: %s", weights)
    logger.info("train_eval.py loaded successfully.")
