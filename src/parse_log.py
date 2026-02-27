"""
parse_log.py
------------
Extract training results from run_grid.py log output.

Usage:
    python parse_log.py                              # uses default paths below
    python parse_log.py --log ../log.txt --out ../results/parsed

Produces:
    <out>/results_summary.csv      — one row per completed run
    <out>/results_summary.json     — same + full epoch history + per-class metrics
    <out>/epoch_history.csv        — epoch-level train_loss / val_macro_f1
    <out>/incomplete_runs.json     — runs that were in-progress when session died

Log line format (Kaggle Papermill):
    <elapsed>s <lineno> <datetime> [LEVEL] <message>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Default paths (run from anywhere)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
DEFAULT_LOG = _HERE.parent / "log.txt"
DEFAULT_OUT = _HERE.parent / "results" / "parsed"

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------
# Strip the Kaggle timing prefix:  "1297.3s 157 "  or just "1297.3s "
_PREFIX = re.compile(r"^\s*[\d.]+s\s+(?:\d+\s+)?")

# The core log message after the prefix + optional datetime + [LEVEL]
_LOG_MSG = re.compile(
    r"(?:(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)\s+\[(\w+)\]\s*)?(.*)",
    re.DOTALL,
)

# [RESULT] line
_RESULT = re.compile(
    r"\[RESULT\]\s+"
    r"type=(\S+)\s+"
    r"comment=(\S+)\s+"
    r"code=(\S+)\s+"
    r"lr=([\de.\-+]+)\s+"
    r"val_f1=([\d.]+)\s+"
    r"test_f1=([\d.]+)"
)

# Epoch header
_EPOCH_HDR = re.compile(r"=== Epoch (\d+) / (\d+) ===")

# Train loss line
_TRAIN_LOSS = re.compile(r"train_loss=([\d.]+)\s+steps=(\d+)")

# Val macro f1
_VAL_F1 = re.compile(r"val_macro_f1=([\d.]+)\s+\(best=([\d.\-]+)\)")

# New best saved
_NEW_BEST = re.compile(r"New best model saved \(epoch (\d+), macro_f1=([\d.]+)\)")

# Classification report row  (class  prec  recall  f1  support)
_REPORT_ROW = re.compile(
    r"^([\w/\-]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$"
)
# summary rows
_REPORT_AVG = re.compile(
    r"^(accuracy|macro avg|weighted avg)\s+([\d.]+)(?:\s+([\d.]+))?(?:\s+([\d.]+))?\s+(\d+)\s*$"
)

# Phase / section headers
_PHASE = re.compile(r"PHASE \d+: (.+)")

# Loading tokenizer / encoder  →  used to detect new run start
_LOADING_ENC = re.compile(r"Loading (?:comment|code) encoder:\s+(\S+)")
_SEED_FIXED = re.compile(r"Seed fixed to (\d+)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_prefix(raw: str) -> str:
    """Remove the Kaggle timing prefix from a log line."""
    return _PREFIX.sub("", raw).strip()


def _parse_classification_report(lines: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Parse a sklearn classification_report block into a dict:
    {
      "Non-SATD": {"precision": .., "recall": .., "f1": .., "support": ..},
      ...
      "accuracy": {"f1": ..., "support": ...},
      "macro avg": {...},
      "weighted avg": {...}
    }
    """
    result: Dict[str, Dict[str, float]] = {}
    for line in lines:
        m = _REPORT_ROW.match(line.strip())
        if m:
            cls, prec, rec, f1, sup = m.groups()
            result[cls] = {
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "support": int(sup),
            }
            continue
        m = _REPORT_AVG.match(line.strip())
        if m:
            label = m.group(1)
            vals = [g for g in m.groups()[1:] if g is not None]
            if label == "accuracy":
                result["accuracy"] = {"f1": float(vals[0]), "support": int(vals[-1])}
            else:
                result[label] = {
                    "precision": float(vals[0]),
                    "recall": float(vals[1]),
                    "f1": float(vals[2]),
                    "support": int(vals[-1]),
                }
    return result


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_log(log_path: Path) -> Dict[str, Any]:
    """
    Parse the full log file and return:
    {
        "completed_runs": [...],   # list of run dicts for [RESULT] lines
        "incomplete_runs": [...],  # runs that started but did not finish
        "phases": [...]            # phase labels encountered
    }
    """
    completed_runs: List[Dict] = []
    phases: List[str] = []

    # State machine for the current run being tracked
    current_run: Dict[str, Any] = {}
    current_epoch: Optional[int] = None
    epoch_history: List[Dict] = []
    in_report: bool = False
    report_lines: List[str] = []
    # which report are we collecting? "val" or "test"
    report_kind: str = ""
    last_result_line: int = 0  # log line index of last [RESULT]
    pending_test_report: bool = False

    raw_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()

    def _flush_report() -> Optional[Dict]:
        if not report_lines:
            return None
        return _parse_classification_report(report_lines)

    for lineno, raw in enumerate(raw_lines):
        msg = _strip_prefix(raw)

        # Extract actual message text (after datetime + [LEVEL])
        m_log = _LOG_MSG.match(msg)
        if m_log:
            text = m_log.group(3).strip()
        else:
            text = msg.strip()

        # ── Phase header ───────────────────────────────────────────────────
        m = _PHASE.search(text)
        if m:
            phases.append(m.group(1).strip())

        # ── Classification report block tracking ───────────────────────────
        # A report block starts with the header line "precision    recall  f1-score   support"
        if re.search(r"precision\s+recall\s+f1-score\s+support", text):
            in_report = True
            report_lines = []
            # Determine if this is a val or test report
            # Test report immediately precedes [RESULT]; val reports come before val_macro_f1
            # We track this by whether we've seen the [RESULT] for the current run yet.
            # After the FINAL epoch val report → next non-empty block is the test report.
            report_kind = "pending"  # will be resolved when we see val_f1 or [RESULT]
            continue

        if in_report:
            stripped = text.strip()
            if stripped == "":
                # blank line inside report — keep accumulating
                continue
            # Check if we've left the report block (new INFO line that is not a report row)
            if (_REPORT_ROW.match(stripped) or _REPORT_AVG.match(stripped) or
                    stripped in ("precision    recall  f1-score   support",)):
                report_lines.append(stripped)
                continue
            # Anything else ends the report
            parsed_report = _flush_report()
            in_report = False
            # Decide where to store it
            if parsed_report:
                if report_kind == "val":
                    if current_epoch is not None and epoch_history:
                        epoch_history[-1]["val_report"] = parsed_report
                elif report_kind == "test":
                    current_run["test_report"] = parsed_report
                else:
                    # Heuristic: if we haven't seen val_f1 for this epoch yet → val
                    # store temporarily
                    current_run.setdefault("_pending_report", parsed_report)
            report_lines = []

        # ── New run start (seed fixed) ─────────────────────────────────────
        m = _SEED_FIXED.search(text)
        if m:
            # Save any in-progress run (should have been [RESULT]'d already)
            current_run = {"epoch_history": epoch_history if epoch_history else []}
            epoch_history = []
            current_epoch = None

        # ── Epoch header ───────────────────────────────────────────────────
        m = _EPOCH_HDR.search(text)
        if m:
            current_epoch = int(m.group(1))
            epoch_history.append({"epoch": current_epoch, "train_loss": None, "val_macro_f1": None})
            report_kind = "val"

        # ── Train loss ─────────────────────────────────────────────────────
        m = _TRAIN_LOSS.search(text)
        if m and epoch_history:
            epoch_history[-1]["train_loss"] = float(m.group(1))
            epoch_history[-1]["steps"] = int(m.group(2))

        # ── Val macro f1 ───────────────────────────────────────────────────
        m = _VAL_F1.search(text)
        if m and epoch_history:
            epoch_history[-1]["val_macro_f1"] = float(m.group(1))
            epoch_history[-1]["previous_best"] = float(m.group(2))
            # Promote pending report to val
            if "_pending_report" in current_run:
                if epoch_history:
                    epoch_history[-1]["val_report"] = current_run.pop("_pending_report")
            report_kind = "test"   # next report will be test

        # ── New best saved ─────────────────────────────────────────────────
        m = _NEW_BEST.search(text)
        if m and epoch_history:
            epoch_history[-1]["is_best"] = True
            epoch_history[-1]["best_macro_f1"] = float(m.group(2))

        # ── [RESULT] line ──────────────────────────────────────────────────
        m = _RESULT.search(text)
        if m:
            model_type, comment_enc, code_enc, lr, val_f1, test_f1 = m.groups()
            # Resolve "-" placeholders
            comment_enc = "" if comment_enc == "-" else comment_enc
            code_enc    = "" if code_enc    == "-" else code_enc

            # Promote _pending_report as test if present
            test_report = current_run.pop("test_report", None)
            if test_report is None and "_pending_report" in current_run:
                test_report = current_run.pop("_pending_report")

            run_record: Dict[str, Any] = {
                "model_type":        model_type,
                "comment_encoder":   comment_enc,
                "code_encoder":      code_enc,
                "lr":                float(lr),
                "best_val_macro_f1": float(val_f1),
                "test_macro_f1":     float(test_f1),
                "epoch_history":     epoch_history[:],
                "test_report":       test_report,
                "log_line":          lineno + 1,
            }

            # Derive best_epoch from history
            best_ep = None
            for eh in epoch_history:
                if eh.get("is_best"):
                    best_ep = eh["epoch"]
            run_record["best_epoch"] = best_ep

            completed_runs.append(run_record)
            last_result_line = lineno

            # Reset for next run
            epoch_history = []
            current_run = {}
            current_epoch = None

    # Anything still in epoch_history after the last [RESULT] = incomplete run
    incomplete: List[Dict] = []
    if epoch_history:
        inc: Dict[str, Any] = {
            "comment_encoder":  current_run.get("comment_encoder", "unknown"),
            "code_encoder":     current_run.get("code_encoder", "unknown"),
            "epoch_history":    epoch_history,
            "last_log_line":    len(raw_lines),
        }
        # Best val so far
        best_val = max(
            (eh["val_macro_f1"] for eh in epoch_history if eh.get("val_macro_f1") is not None),
            default=None,
        )
        inc["best_val_macro_f1_so_far"] = best_val
        incomplete.append(inc)

    return {
        "completed_runs": completed_runs,
        "incomplete_runs": incomplete,
        "phases": phases,
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def build_summary_df(completed_runs: List[Dict]) -> pd.DataFrame:
    """Flatten completed runs to a tidy DataFrame."""
    rows = []
    labels = ["Non-SATD", "Design", "Requirement", "Defect", "Test", "Documentation"]
    for r in completed_runs:
        row: Dict[str, Any] = {
            "model_type":        r["model_type"],
            "comment_encoder":   r["comment_encoder"],
            "code_encoder":      r["code_encoder"],
            "lr":                r["lr"],
            "best_val_macro_f1": r["best_val_macro_f1"],
            "test_macro_f1":     r["test_macro_f1"],
            "best_epoch":        r.get("best_epoch"),
            "log_line":          r.get("log_line"),
        }
        # Per-class test f1 scores
        tr = r.get("test_report") or {}
        for lbl in labels:
            row[f"test_f1_{lbl}"]       = (tr.get(lbl) or {}).get("f1")
            row[f"test_prec_{lbl}"]     = (tr.get(lbl) or {}).get("precision")
            row[f"test_recall_{lbl}"]   = (tr.get(lbl) or {}).get("recall")
            row[f"test_support_{lbl}"]  = (tr.get(lbl) or {}).get("support")
        row["test_accuracy"]         = (tr.get("accuracy") or {}).get("f1")
        row["test_macro_f1_report"]  = (tr.get("macro avg") or {}).get("f1")
        rows.append(row)
    return pd.DataFrame(rows)


def build_epoch_df(completed_runs: List[Dict]) -> pd.DataFrame:
    rows = []
    for r in completed_runs:
        run_id = f"{r['model_type']}|{r['comment_encoder']}|{r['code_encoder']}|lr={r['lr']:.0e}"
        for eh in r.get("epoch_history", []):
            rows.append({
                "run_id":          run_id,
                "model_type":      r["model_type"],
                "comment_encoder": r["comment_encoder"],
                "code_encoder":    r["code_encoder"],
                "lr":              r["lr"],
                "epoch":           eh.get("epoch"),
                "train_loss":      eh.get("train_loss"),
                "val_macro_f1":    eh.get("val_macro_f1"),
                "is_best":         bool(eh.get("is_best", False)),
            })
    return pd.DataFrame(rows)


def results_to_json(parsed: Dict) -> List[Dict]:
    """Convert parsed results to JSON-serialisable dicts."""
    out = []
    for r in parsed["completed_runs"]:
        jr = {k: v for k, v in r.items() if k not in ("state_dict",)}
        out.append(jr)
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Extract results from run_grid.py log.")
    ap.add_argument("--log", default=str(DEFAULT_LOG), help="Path to log.txt")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="Output directory")
    args = ap.parse_args(argv)

    log_path = Path(args.log)
    out_dir  = Path(args.out)

    if not log_path.exists():
        print(f"ERROR: Log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Parsing log: {log_path}")
    parsed = parse_log(log_path)
    completed = parsed["completed_runs"]
    incomplete = parsed["incomplete_runs"]

    print(f"\n{'='*60}")
    print(f"Completed runs : {len(completed)}")
    print(f"Incomplete runs: {len(incomplete)}")
    print(f"Phases detected: {parsed['phases']}")
    print(f"{'='*60}\n")

    if not completed:
        print("No [RESULT] lines found. Nothing to save.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Summary CSV ─────────────────────────────────────────────────────────
    summary_df = build_summary_df(completed)
    summary_csv = out_dir / "results_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}")

    # ── Epoch history CSV ────────────────────────────────────────────────────
    epoch_df = build_epoch_df(completed)
    epoch_csv = out_dir / "epoch_history.csv"
    epoch_df.to_csv(epoch_csv, index=False)
    print(f"Saved: {epoch_csv}")

    # ── Full JSON ─────────────────────────────────────────────────────────────
    json_path = out_dir / "results_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_to_json(parsed), f, indent=2)
    print(f"Saved: {json_path}")

    # ── Incomplete runs JSON ──────────────────────────────────────────────────
    if incomplete:
        inc_path = out_dir / "incomplete_runs.json"
        with open(inc_path, "w", encoding="utf-8") as f:
            json.dump(incomplete, f, indent=2)
        print(f"Saved: {inc_path}")

    # ── Console summary table ─────────────────────────────────────────────────
    print("\n── Completed runs (sorted by test_macro_f1 desc) ──")
    display_cols = ["model_type", "comment_encoder", "code_encoder", "lr",
                    "best_val_macro_f1", "test_macro_f1", "best_epoch"]
    disp = summary_df[display_cols].copy()
    disp["lr"] = disp["lr"].map(lambda x: f"{x:.0e}")
    disp = disp.sort_values("test_macro_f1", ascending=False)
    # Use pandas to_string for clean output
    pd.set_option("display.max_colwidth", 40)
    pd.set_option("display.width", 160)
    print(disp.to_string(index=False))

    if incomplete:
        print("\n── Incomplete / in-progress runs at session timeout ──")
        for inc in incomplete:
            eh = inc.get("epoch_history", [])
            best_val = inc.get("best_val_macro_f1_so_far")
            completed_epochs = [e["epoch"] for e in eh if e.get("val_macro_f1") is not None]
            print(
                f"  comment={inc['comment_encoder'] or '?'}  "
                f"code={inc['code_encoder'] or '?'}  "
                f"epochs_done={completed_epochs}  "
                f"best_val_f1_so_far={best_val}"
            )

    print(f"\nDone. All outputs in: {out_dir}")


if __name__ == "__main__":
    main()
