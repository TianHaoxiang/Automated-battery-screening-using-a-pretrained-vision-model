#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monitor a running AMOTF-NPZ training job until it finishes, then write a comparison report
against historical CMAOTN baselines.

This script is designed to run unattended (e.g. via nohup) and does not require secrets.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _last_epoch_from_log(text: str) -> Optional[int]:
    best: Optional[int] = None
    for ln in text.splitlines():
        if " epoch=" not in ln:
            continue
        m = re.search(r"epoch=(\d+)", ln)
        if m:
            best = int(m.group(1))
    return best


def _load_json(p: Path) -> Dict[str, Any]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _find_latest_sweep_step_dir(sweep_root: Path) -> Optional[Path]:
    if not sweep_root.is_dir():
        return None
    step_dirs = [p for p in sweep_root.iterdir() if p.is_dir() and re.match(r"^s\\d\\d_", p.name)]
    if not step_dirs:
        return None
    step_dirs.sort(key=lambda p: p.name)
    return step_dirs[-1]


def _parse_dataset_sweep_last_summary(sweep_root: Path) -> Tuple[Optional[Path], Dict[str, Any]]:
    results = sweep_root / "dataset_sweep_results.csv"
    if results.is_file():
        try:
            with open(results, "r", encoding="utf-8", errors="replace", newline="") as f:
                rows = list(csv.DictReader(f))
            if rows:
                last = rows[-1]
                run_dir = Path(str(last.get("run_dir", "")).strip())
                summ = run_dir / "cmaotn_npz" / "summary.json"
                if summ.is_file():
                    return summ, _load_json(summ)
        except Exception:
            pass
    last_dir = _find_latest_sweep_step_dir(sweep_root)
    if last_dir is None:
        return None, {}
    summ = last_dir / "cmaotn_npz" / "summary.json"
    return (summ if summ.is_file() else None), (_load_json(summ) if summ.is_file() else {})


def _metric_triplet(d: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    def _get(k: str) -> Optional[float]:
        try:
            v = d.get(k, None)
            return None if v is None else float(v)
        except Exception:
            return None

    return _get("test_acc_mean"), _get("test_macro_f1_mean"), _get("test_weighted_f1_mean")


def _fmt(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    return f"{v:.4f}"


def _write_report(
    *,
    out_path: Path,
    amotf_summary: Dict[str, Any],
    cmaotn_full_summary: Dict[str, Any],
    cmaotn_sweep_last_summary: Dict[str, Any],
    extra: Dict[str, Any],
) -> None:
    am_acc, am_mf1, am_wf1 = _metric_triplet(amotf_summary)
    cf_acc, cf_mf1, cf_wf1 = _metric_triplet(cmaotn_full_summary)
    sw_acc, sw_mf1, sw_wf1 = _metric_triplet(cmaotn_sweep_last_summary)

    def _diff(a: Optional[float], b: Optional[float]) -> str:
        if a is None or b is None:
            return "N/A"
        return f"{(a - b):+.4f}"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# AMOTF vs CMAOTN Comparison Report")
    lines.append("")
    lines.append("## Run Info")
    for k in ["run_dir", "train_log", "epochs_expected", "last_epoch_seen", "finished_detected_at"]:
        if k in extra:
            lines.append(f"- {k}: {extra[k]}")
    lines.append("")
    lines.append("## Key Metrics (test mean)")
    lines.append("")
    lines.append("| Metric | AMOTF (this run) | CMAOTN full_run_all | Diff | CMAOTN sweep last | Diff |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    lines.append(
        f"| acc | {_fmt(am_acc)} | {_fmt(cf_acc)} | {_diff(am_acc, cf_acc)} | {_fmt(sw_acc)} | {_diff(am_acc, sw_acc)} |"
    )
    lines.append(
        f"| macro_f1 | {_fmt(am_mf1)} | {_fmt(cf_mf1)} | {_diff(am_mf1, cf_mf1)} | {_fmt(sw_mf1)} | {_diff(am_mf1, sw_mf1)} |"
    )
    lines.append(
        f"| weighted_f1 | {_fmt(am_wf1)} | {_fmt(cf_wf1)} | {_diff(am_wf1, cf_wf1)} | {_fmt(sw_wf1)} | {_diff(am_wf1, sw_wf1)} |"
    )
    lines.append("")
    lines.append("## Notes")
    lines.append("- This report compares *aggregate test means* when available. For single-fold runs, the mean equals the fold metric.")
    lines.append("- For deeper inspection, open the corresponding `fold_metrics.csv` and per-fold artifacts under each run directory.")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Path to AMOTF run_dir (e.g. .../amotf_npz)")
    ap.add_argument("--poll_sec", type=int, default=300)
    ap.add_argument("--epochs_expected", type=int, default=50)
    ap.add_argument("--cmaotn_full_summary", required=True, help="Path to historical CMAOTN full_run_all summary.json")
    ap.add_argument("--cmaotn_sweep_root", required=True, help="Path to historical CMAOTN dataset_sweep root dir")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    train_log = run_dir / "train.log"
    summary = run_dir / "summary.json"
    report = run_dir / "comparison_report.md"

    cma_full = _load_json(Path(args.cmaotn_full_summary))
    _, cma_sweep_last = _parse_dataset_sweep_last_summary(Path(args.cmaotn_sweep_root))

    while True:
        txt = _read_text(train_log)
        last_ep = _last_epoch_from_log(txt)
        finished = bool(summary.is_file())
        if finished:
            break
        # If we already saw the last epoch line but summary isn't written yet, wait a bit longer.
        if last_ep is not None and int(last_ep) >= int(args.epochs_expected) - 1:
            time.sleep(30)
        else:
            time.sleep(max(5, int(args.poll_sec)))

    am = _load_json(summary)
    extra = {
        "run_dir": str(run_dir),
        "train_log": str(train_log),
        "epochs_expected": int(args.epochs_expected),
        "last_epoch_seen": last_ep,
        "finished_detected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _write_report(
        out_path=report,
        amotf_summary=am,
        cmaotn_full_summary=cma_full,
        cmaotn_sweep_last_summary=cma_sweep_last,
        extra=extra,
    )
    print("wrote", str(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

