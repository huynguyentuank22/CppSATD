"""
parse_phase3.py
---------------
Extract results (complete + partial) from the Phase-3 Kaggle log file.

Usage:
    python parse_phase3.py
    python parse_phase3.py --log ../results/phase3.txt --out ../results/phase3

Outputs in <out>/:
    run_summary.json        — full structured results (config + epoch history)
    run_summary.csv         — flat table (one row per run attempt)
    epoch_history.csv       — epoch-level metrics for every run
    data_split.json         — dataset / label distribution info extracted from log
    errors.json             — timeout / exception info detected in the log

Log format (Kaggle Papermill):
    <elapsed>s <lineno> <datetime> [LEVEL] <message>
    OR just plain lines for tracebacks / progress bars.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Paths (relative to this file)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
DEFAULT_LOG = _HERE.parent / "results" / "phase3.txt"
DEFAULT_OUT = _HERE.parent / "results" / "phase3"

# ---------------------------------------------------------------------------
# Regexes
# ---------------------------------------------------------------------------
_PREFIX     = re.compile(r"^\s*[\d.]+s\s+(?:\d+\s+)?")
_LOG_MSG    = re.compile(r"(?:\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+\s+\[(\w+)\]\s*)?(.*)", re.DOTALL)

# Dataset / split info
_NORMALISED = re.compile(r"Normalised dataset:\s*(\d+)\s*rows")
_LABEL_DIST = re.compile(r"^(\w[\w/\-]*)\s+(\d+)$")
_SPLIT_LINE = re.compile(r"Train\s+(\d+)\s+\|\s+Val\s+(\d+)\s+\|\s+Test\s+(\d+)")
_SPLIT_DIST = re.compile(r"(Train|Val|Test)\s+dist:\s+(\{.+\})")
_CLASS_W    = re.compile(r"Class weights:\s*(\[[\d.,\s]+\])")
_CLASS_CNT  = re.compile(r"Class counts:\s*(\[[\d.,\s]+\])")

# Model / run config
_PHASE_HDR  = re.compile(r"PHASE \d+:\s*(.+)")
_LATE_FUSE  = re.compile(r"LateFusion\s+\|\s+comment=(\S+)\s+code=(\S+)")
_SEED       = re.compile(r"Seed fixed to (\d+)")
_LR_CFG     = re.compile(r"lr\s*=\s*([\d.e\-+]+)")
_BATCH_CFG  = re.compile(r"batch(?:_size)?\s*=\s*(\d+)")
_EPOCH_CFG  = re.compile(r"n_epochs?\s*=\s*(\d+)")

# Training progress
_EPOCH_HDR  = re.compile(r"=== Epoch (\d+) / (\d+) ===")
_TRAIN_LOSS = re.compile(r"train_loss=([\d.]+)\s+steps=(\d+)")
_VAL_F1     = re.compile(r"val_macro_f1=([\d.]+)\s+\(best=([\d.\-]+)\)")
_NEW_BEST   = re.compile(r"New best model saved \(epoch (\d+), macro_f1=([\d.]+)\)")
_RESULT     = re.compile(
    r"\[RESULT\]\s+type=(\S+)\s+comment=(\S+)\s+code=(\S+)\s+"
    r"lr=([\de.\-+]+)\s+val_f1=([\d.]+)\s+test_f1=([\d.]+)"
)
_RPT_HDR    = re.compile(r"precision\s+recall\s+f1-score\s+support")
_RPT_ROW    = re.compile(r"^([\w/\-]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$")
_RPT_AVG    = re.compile(
    r"^(accuracy|macro avg|weighted avg)\s+([\d.]+)(?:\s+([\d.]+))?(?:\s+([\d.]+))?\s+(\d+)\s*$"
)

# Errors / timeouts
_TIMEOUT    = re.compile(r"Timeout waiting for execute reply\s*\((\d+)s\)")
_CELL_TOUT  = re.compile(r"CellTimeoutError.*after (\d+) seconds")
_KEYBOARD   = re.compile(r"KeyboardInterrupt")
_TRACEBACK  = re.compile(r"^Traceback \(most recent call last\)")
_ERR_FILE   = re.compile(r'File "([^"]+)", line (\d+), in (\S+)')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip(raw: str) -> str:
    return _PREFIX.sub("", raw).strip()


def _text(raw: str) -> str:
    """Return the pure message text, stripping prefix + optional datetime/level."""
    msg = _strip(raw)
    m = _LOG_MSG.match(msg)
    return m.group(2).strip() if m else msg.strip()


def _parse_clf_report(lines: List[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for line in lines:
        s = line.strip()
        m = _RPT_ROW.match(s)
        if m:
            cls, prec, rec, f1, sup = m.groups()
            out[cls] = {"precision": float(prec), "recall": float(rec),
                        "f1": float(f1), "support": int(sup)}
            continue
        m = _RPT_AVG.match(s)
        if m:
            label = m.group(1)
            vals = [g for g in m.groups()[1:] if g is not None]
            if label == "accuracy":
                out["accuracy"] = {"f1": float(vals[0]), "support": int(vals[-1])}
            else:
                out[label] = {"precision": float(vals[0]), "recall": float(vals[1]),
                              "f1": float(vals[2]), "support": int(vals[-1])}
    return out


def _safe_list(s: str) -> List[float]:
    """Parse '[0.01, 0.026]' → [0.01, 0.026]."""
    return [float(x) for x in re.findall(r"[\d.eE+\-]+", s)]


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_phase3_log(log_path: Path) -> Dict[str, Any]:
    """
    Parse phase3.txt and return a structured dict:
    {
      "data_info":       { rows, label_distribution, splits, class_weights, ... },
      "phase":           str,
      "completed_runs":  [ { model_type, comment_encoder, code_encoder, lr,
                              best_val_macro_f1, test_macro_f1, best_epoch,
                              epoch_history, test_report } ],
      "incomplete_runs": [ { comment_encoder, code_encoder, lr, epoch_history,
                              best_val_macro_f1_so_far, stopped_at_epoch } ],
      "errors":          [ { kind, detail, traceback } ],
    }
    """
    raw_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()

    # ── state ──────────────────────────────────────────────────────────────
    data_info:       Dict[str, Any] = {}
    phase_label:     str            = ""
    completed_runs:  List[Dict]     = []
    incomplete_runs: List[Dict]     = []
    errors:          List[Dict]     = []

    # per-run state
    cur_run:       Dict[str, Any]       = {}
    epoch_history: List[Dict[str, Any]] = []
    in_report:     bool                 = False
    report_lines:  List[str]            = []
    report_kind:   str                  = ""
    in_tb:         bool                 = False
    tb_lines:      List[str]            = []
    tb_kind:       str                  = ""

    # collecting label distribution (multi-line)
    collecting_labels: bool = False
    label_block:       List[str] = []

    for raw in raw_lines:
        t = _text(raw)

        # ── Traceback collection ────────────────────────────────────────────
        if _TRACEBACK.match(t):
            # flush previous if any
            if in_tb and tb_lines:
                errors.append({"kind": tb_kind or "Traceback", "detail": "", "traceback": tb_lines[:]})
            in_tb = True
            tb_lines = [t]
            tb_kind = "Traceback"
            continue
        if in_tb:
            tb_lines.append(t)
            m = _ERR_FILE.match(t)
            # end of traceback = line that does not look like a tb frame
            # (empty, or starts a new section, etc.) — we keep accumulating
            # until we see a named exception or a new [INFO] line picks up
            if re.match(r'^[A-Z][\w]+Error|KeyboardInterrupt|CellTimeoutError', t):
                tb_kind = t.split(":")[0].strip()
                errors.append({"kind": tb_kind, "detail": t, "traceback": tb_lines[:]})
                in_tb = False
                tb_lines = []

        # ── Timeout markers ────────────────────────────────────────────────
        m = _TIMEOUT.search(t)
        if m:
            errors.append({
                "kind": "KaggleTimeout",
                "detail": f"Timeout waiting for execute reply after {m.group(1)}s",
                "traceback": []
            })

        m = _CELL_TOUT.search(t)
        if m:
            errors.append({
                "kind": "CellTimeoutError",
                "detail": f"Cell timed out after {m.group(1)}s",
                "traceback": []
            })

        if _KEYBOARD.search(t) and not in_tb:
            errors.append({"kind": "KeyboardInterrupt", "detail": t, "traceback": []})

        # ── Data / split info ──────────────────────────────────────────────
        m = _NORMALISED.search(t)
        if m:
            data_info["total_rows"] = int(m.group(1))

        if "Label distribution:" in t:
            collecting_labels = True
            label_block = []
            continue
        if collecting_labels:
            m = _LABEL_DIST.match(t)
            if m:
                label_block.append((m.group(1), int(m.group(2))))
            elif t == "" and label_block:
                pass  # blank line mid-block is fine
            else:
                if label_block:
                    data_info["label_distribution"] = dict(label_block)
                collecting_labels = False

        m = _SPLIT_LINE.search(t)
        if m:
            data_info["split_sizes"] = {
                "train": int(m.group(1)),
                "val":   int(m.group(2)),
                "test":  int(m.group(3)),
            }

        m = _SPLIT_DIST.search(t)
        if m:
            key = m.group(1).lower() + "_dist"
            try:
                data_info[key] = json.loads(m.group(2).replace("'", '"'))
            except Exception:
                data_info[key] = m.group(2)

        m = _CLASS_CNT.search(t)
        if m:
            data_info["class_counts"] = _safe_list(m.group(1))

        m = _CLASS_W.search(t)
        if m:
            data_info["class_weights"] = _safe_list(m.group(1))

        # ── Phase header ───────────────────────────────────────────────────
        m = _PHASE_HDR.search(t)
        if m:
            phase_label = m.group(1).strip()

        # ── Run start (LateFusion config line) ─────────────────────────────
        m = _LATE_FUSE.search(t)
        if m:
            # If a prior run was in-progress, save it as incomplete
            if epoch_history:
                _save_incomplete(cur_run, epoch_history, incomplete_runs)
            cur_run = {
                "model_type":      "LateFusion",
                "comment_encoder": m.group(1),
                "code_encoder":    m.group(2),
            }
            epoch_history = []
            in_report = False
            report_lines = []

        # ── Hyperparams (pick up lr / batch / epochs from log) ─────────────
        if cur_run:
            for pat, key in ((_LR_CFG, "lr"), (_BATCH_CFG, "batch_size"), (_EPOCH_CFG, "n_epochs")):
                m = pat.search(t)
                if m and key not in cur_run:
                    try:
                        cur_run[key] = float(m.group(1)) if key == "lr" else int(m.group(1))
                    except ValueError:
                        pass

        # ── Classification report block ────────────────────────────────────
        if _RPT_HDR.search(t):
            in_report = True
            report_lines = []
            report_kind = "pending"
            continue

        if in_report:
            s = t.strip()
            if _RPT_ROW.match(s) or _RPT_AVG.match(s) or s == "":
                if s:
                    report_lines.append(s)
                continue
            # Non-report line ends the block
            parsed_rpt = _parse_clf_report(report_lines)
            in_report = False
            report_lines = []
            if parsed_rpt:
                if report_kind == "val" and epoch_history:
                    epoch_history[-1]["val_report"] = parsed_rpt
                elif report_kind == "test":
                    cur_run["test_report"] = parsed_rpt
                else:
                    cur_run["_pending_report"] = parsed_rpt

        # ── Epoch header ───────────────────────────────────────────────────
        m = _EPOCH_HDR.search(t)
        if m:
            ep, total = int(m.group(1)), int(m.group(2))
            cur_run["total_epochs"] = total
            epoch_history.append({
                "epoch":         ep,
                "total_epochs":  total,
                "train_loss":    None,
                "val_macro_f1":  None,
            })
            report_kind = "val"

        # ── Train loss ─────────────────────────────────────────────────────
        m = _TRAIN_LOSS.search(t)
        if m and epoch_history:
            epoch_history[-1]["train_loss"] = float(m.group(1))
            epoch_history[-1]["steps"]      = int(m.group(2))

        # ── Val macro F1 ───────────────────────────────────────────────────
        m = _VAL_F1.search(t)
        if m and epoch_history:
            epoch_history[-1]["val_macro_f1"]    = float(m.group(1))
            epoch_history[-1]["previous_best_f1"] = float(m.group(2))
            if "_pending_report" in cur_run:
                epoch_history[-1]["val_report"] = cur_run.pop("_pending_report")
            report_kind = "test"

        # ── New best ───────────────────────────────────────────────────────
        m = _NEW_BEST.search(t)
        if m and epoch_history:
            epoch_history[-1]["is_best"]        = True
            epoch_history[-1]["best_macro_f1"]  = float(m.group(2))

        # ── [RESULT] line (completed run) ──────────────────────────────────
        m = _RESULT.search(t)
        if m:
            model_type, c_enc, k_enc, lr, val_f1, test_f1 = m.groups()
            test_report = cur_run.pop("test_report", None)
            if test_report is None:
                test_report = cur_run.pop("_pending_report", None)

            best_ep = next(
                (eh["epoch"] for eh in reversed(epoch_history) if eh.get("is_best")),
                None
            )

            completed_runs.append({
                "model_type":        model_type,
                "comment_encoder":   "" if c_enc == "-" else c_enc,
                "code_encoder":      "" if k_enc == "-" else k_enc,
                "lr":                float(lr),
                "best_val_macro_f1": float(val_f1),
                "test_macro_f1":     float(test_f1),
                "best_epoch":        best_ep,
                "epoch_history":     epoch_history[:],
                "test_report":       test_report,
            })
            epoch_history = []
            cur_run = {}

    # ── Flush any remaining in-progress run ────────────────────────────────
    if epoch_history:
        _save_incomplete(cur_run, epoch_history, incomplete_runs)

    return {
        "phase":           phase_label,
        "data_info":       data_info,
        "completed_runs":  completed_runs,
        "incomplete_runs": incomplete_runs,
        "errors":          errors,
    }


def _save_incomplete(
    cur_run: Dict[str, Any],
    epoch_history: List[Dict[str, Any]],
    incomplete_runs: List[Dict],
) -> None:
    best_val = max(
        (eh["val_macro_f1"] for eh in epoch_history if eh.get("val_macro_f1") is not None),
        default=None,
    )
    epochs_done = [eh["epoch"] for eh in epoch_history if eh.get("val_macro_f1") is not None]
    incomplete_runs.append({
        "model_type":             cur_run.get("model_type", "unknown"),
        "comment_encoder":        cur_run.get("comment_encoder", "unknown"),
        "code_encoder":           cur_run.get("code_encoder", "unknown"),
        "lr":                     cur_run.get("lr"),
        "total_epochs_planned":   cur_run.get("total_epochs"),
        "batch_size":             cur_run.get("batch_size"),
        "epochs_completed":       epochs_done,
        "best_val_macro_f1_so_far": best_val,
        "epoch_history":          epoch_history,
    })


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def build_run_df(completed: List[Dict], incomplete: List[Dict]) -> pd.DataFrame:
    labels = ["Non-SATD", "Design", "Requirement", "Defect", "Test", "Documentation"]
    rows = []
    for r in completed:
        tr = r.get("test_report") or {}
        row: Dict[str, Any] = {
            "status":            "completed",
            "model_type":        r["model_type"],
            "comment_encoder":   r["comment_encoder"],
            "code_encoder":      r["code_encoder"],
            "lr":                r.get("lr"),
            "best_val_macro_f1": r["best_val_macro_f1"],
            "test_macro_f1":     r["test_macro_f1"],
            "best_epoch":        r.get("best_epoch"),
            "test_accuracy":     (tr.get("accuracy") or {}).get("f1"),
            "test_macro_avg_f1": (tr.get("macro avg") or {}).get("f1"),
            "test_weighted_f1":  (tr.get("weighted avg") or {}).get("f1"),
        }
        for lbl in labels:
            row[f"f1_{lbl}"] = (tr.get(lbl) or {}).get("f1")
        rows.append(row)

    for r in incomplete:
        row = {
            "status":            "incomplete (timeout)",
            "model_type":        r.get("model_type", ""),
            "comment_encoder":   r.get("comment_encoder", ""),
            "code_encoder":      r.get("code_encoder", ""),
            "lr":                r.get("lr"),
            "best_val_macro_f1": r.get("best_val_macro_f1_so_far"),
            "test_macro_f1":     None,
            "best_epoch":        None,
            "test_accuracy":     None,
            "test_macro_avg_f1": None,
            "test_weighted_f1":  None,
        }
        for lbl in labels:
            row[f"f1_{lbl}"] = None
        rows.append(row)

    return pd.DataFrame(rows)


def build_epoch_df(completed: List[Dict], incomplete: List[Dict]) -> pd.DataFrame:
    rows = []
    for src, status in [(completed, "completed"), (incomplete, "incomplete")]:
        for r in src:
            run_id = (
                f"{r.get('model_type','')}|"
                f"{r.get('comment_encoder','')}|"
                f"{r.get('code_encoder','')}|"
                f"lr={r.get('lr', '?')}"
            )
            for eh in r.get("epoch_history", []):
                rows.append({
                    "run_id":          run_id,
                    "status":          status,
                    "comment_encoder": r.get("comment_encoder", ""),
                    "code_encoder":    r.get("code_encoder", ""),
                    "lr":              r.get("lr"),
                    "epoch":           eh.get("epoch"),
                    "total_epochs":    eh.get("total_epochs"),
                    "train_loss":      eh.get("train_loss"),
                    "val_macro_f1":    eh.get("val_macro_f1"),
                    "is_best":         bool(eh.get("is_best", False)),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Extract Phase-3 results from Kaggle log.")
    ap.add_argument("--log", default=str(DEFAULT_LOG), help="Path to phase3.txt")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="Output directory")
    args = ap.parse_args(argv)

    log_path = Path(args.log)
    out_dir  = Path(args.out)

    if not log_path.exists():
        print(f"ERROR: Log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Parsing: {log_path}  ({log_path.stat().st_size / 1024:.1f} KB, "
          f"{sum(1 for _ in log_path.open(encoding='utf-8', errors='replace'))} lines)")

    parsed = parse_phase3_log(log_path)

    completed  = parsed["completed_runs"]
    incomplete = parsed["incomplete_runs"]
    errors     = parsed["errors"]

    # ── Console summary ─────────────────────────────────────────────────────
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  Phase          : {parsed['phase']}")
    print(f"  Completed runs : {len(completed)}")
    print(f"  Incomplete runs: {len(incomplete)}")
    print(f"  Errors detected: {len(errors)}")
    print(sep)

    di = parsed["data_info"]
    if di:
        print("\n── Dataset ──")
        print(f"  Total rows : {di.get('total_rows', '?')}")
        ss = di.get("split_sizes", {})
        if ss:
            print(f"  Train / Val / Test : {ss.get('train')} / {ss.get('val')} / {ss.get('test')}")
        ld = di.get("label_distribution", {})
        if ld:
            print("  Label distribution:")
            for lbl, cnt in ld.items():
                print(f"    {lbl:<20} {cnt:>6}")
        cw = di.get("class_weights", [])
        if cw:
            label_names = ["Non-SATD", "Design", "Requirement", "Defect", "Test", "Documentation"]
            print("  Class weights (for loss):")
            for lbl, w in zip(label_names, cw):
                print(f"    {lbl:<20} {w:.4f}")

    if completed:
        print("\n── Completed runs ──")
        for r in completed:
            print(
                f"  [{r['model_type']}]  comment={r['comment_encoder']}  "
                f"code={r['code_encoder']}  lr={r.get('lr', '?'):.0e}  "
                f"val_f1={r['best_val_macro_f1']:.4f}  test_f1={r['test_macro_f1']:.4f}  "
                f"best_epoch={r.get('best_epoch', '?')}"
            )
    else:
        print("\n  [!] No completed runs found (session likely timed out).")

    if incomplete:
        print("\n── Incomplete / timed-out runs ──")
        for r in incomplete:
            eh = r.get("epoch_history", [])
            epochs_done = [e["epoch"] for e in eh if e.get("val_macro_f1") is not None]
            epochs_started = [e["epoch"] for e in eh]
            print(
                f"  [{r.get('model_type', '?')}]  "
                f"comment={r.get('comment_encoder', '?')}  "
                f"code={r.get('code_encoder', '?')}  "
                f"lr={r.get('lr', '?')}  "
                f"total_epochs_planned={r.get('total_epochs_planned', '?')}  "
                f"epochs_started={epochs_started}  "
                f"epochs_with_val_f1={epochs_done}  "
                f"best_val_f1_so_far={r.get('best_val_macro_f1_so_far', 'N/A')}"
            )
            for e in eh:
                print(
                    f"    epoch {e['epoch']}: "
                    f"train_loss={e.get('train_loss', 'N/A')}  "
                    f"val_f1={e.get('val_macro_f1', 'N/A')}"
                )

    if errors:
        print("\n── Errors / Timeouts ──")
        seen_kinds = set()
        for err in errors:
            k = err["kind"]
            if k not in seen_kinds:
                print(f"  [{k}] {err['detail'][:120]}")
                seen_kinds.add(k)

    # ── Save outputs ────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)

    # run_summary.json
    summary_json = out_dir / "run_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2, default=str)
    print(f"\nSaved: {summary_json}")

    # run_summary.csv
    run_df = build_run_df(completed, incomplete)
    if not run_df.empty:
        run_csv = out_dir / "run_summary.csv"
        run_df.to_csv(run_csv, index=False)
        print(f"Saved: {run_csv}")

    # epoch_history.csv
    ep_df = build_epoch_df(completed, incomplete)
    if not ep_df.empty:
        ep_csv = out_dir / "epoch_history.csv"
        ep_df.to_csv(ep_csv, index=False)
        print(f"Saved: {ep_csv}")

    # data_split.json
    if di:
        ds_json = out_dir / "data_split.json"
        with open(ds_json, "w", encoding="utf-8") as f:
            json.dump(di, f, indent=2)
        print(f"Saved: {ds_json}")

    # errors.json
    if errors:
        err_json = out_dir / "errors.json"
        with open(err_json, "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2, default=str)
        print(f"Saved: {err_json}")

    print(f"\nDone. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
