#!/usr/bin/env python3
"""Build the validation-locked dense-field AMOTF v2 representation.

The numerical construction follows Cursor's locked ``dense_field_m5`` builder:
ordinal-pattern transition fields are mixed with signed amplitude and pairwise
EMD fields at weights 0.50/0.25/0.25.  Outputs are written beside each sample
under a separate encoding directory so legacy sparse AMOTF files are preserved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


DATASETS = ["CALCE", "HNEI", "Oxford", "SNL_LFP", "SNL_NCA", "SNL_NMC", "UL_Purdue"]


def resolve_path(path: str, data_root: str) -> Path:
    text = str(path or "").strip().replace("\\", "/")
    if re.match(r"^[A-Za-z]:/", text):
        text = re.sub(r"^[A-Za-z]:/", "", text)
    candidate = Path(text)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    return Path(data_root) / text.lstrip("/")


def _pick_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lower = {str(column).strip().lower(): str(column) for column in df.columns}
    for name in candidates:
        if name in lower:
            return lower[name]
    for key, column in lower.items():
        if any(name in key for name in candidates):
            return column
    return None


def read_curve_csv(curve_csv: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(curve_csv)
    except Exception:
        return None
    voltage_col = _pick_column(df, ["voltage", "voltage/v", "voltage_v", "volt"])
    current_col = _pick_column(df, ["current", "current/a", "current_a", "curr"])
    soc_col = _pick_column(df, ["soc_pct", "soc", "state_of_charge", "state of charge"])
    phase_col = _pick_column(df, ["phase", "status", "mode", "step_type"])
    if voltage_col is None or current_col is None:
        return None
    out = pd.DataFrame(
        {
            "voltage": pd.to_numeric(df[voltage_col], errors="coerce"),
            "current": pd.to_numeric(df[current_col], errors="coerce"),
        }
    )
    out["soc"] = (
        pd.to_numeric(df[soc_col], errors="coerce")
        if soc_col is not None
        else np.linspace(0.0, 1.0, len(out), dtype=float)
    )
    out["phase"] = df[phase_col].astype(str) if phase_col is not None else ""
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["voltage", "current", "soc"])
    return None if len(out) < 16 else out.reset_index(drop=True)


def split_charge_discharge(curve: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    phase = curve["phase"].astype(str).str.lower()
    charge_mask = phase.str.contains("charge") & ~phase.str.contains("discharge")
    discharge_mask = phase.str.contains("discharge")
    if int(charge_mask.sum()) < 8 or int(discharge_mask.sum()) < 8:
        current = curve["current"].to_numpy(dtype=float)
        charge_mask = pd.Series(current > 0, index=curve.index)
        discharge_mask = pd.Series(current < 0, index=curve.index)
    if int(charge_mask.sum()) < 8 or int(discharge_mask.sum()) < 8:
        voltage = curve["voltage"].to_numpy(dtype=float)
        split = int(np.nanargmax(voltage))
        if split < 8 or len(curve) - split - 1 < 8:
            return None, None
        charge_mask = pd.Series(False, index=curve.index)
        discharge_mask = pd.Series(False, index=curve.index)
        charge_mask.iloc[: split + 1] = True
        discharge_mask.iloc[split + 1 :] = True
    return curve.loc[charge_mask].copy(), curve.loc[discharge_mask].copy()


def resample_by_soc(soc: np.ndarray, values: np.ndarray, size: int) -> np.ndarray:
    x = np.asarray(soc, dtype=float)
    y = np.asarray(values, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size == 0:
        return np.zeros(int(size), dtype=np.float32)
    reverse_output = False
    if x.size > 1 and float(x[0]) > float(x[-1]):
        x, y = x[::-1], y[::-1]
        reverse_output = True
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    unique_x = np.unique(x)
    if unique_x.size != x.size:
        y = np.asarray([float(np.mean(y[x == value])) for value in unique_x], dtype=float)
        x = unique_x
    if x.size == 1:
        return np.full(int(size), float(y[0]), dtype=np.float32)
    query = np.linspace(float(x[0]), float(x[-1]), int(size), dtype=float)
    output = np.interp(query, x, y).astype(np.float32)
    return output[::-1].copy() if reverse_output else output


def normalize_minmax(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    lo = float(np.nanmin(arr)) if arr.size else 0.0
    hi = float(np.nanmax(arr)) if arr.size else 0.0
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1.0e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def normalize_maxabs(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    scale = float(np.nanmax(np.abs(arr))) if arr.size else 0.0
    if not np.isfinite(scale) or scale < 1.0e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / scale).astype(np.float32)


def permutation_to_index(permutation: np.ndarray) -> int:
    code = 0
    m = int(len(permutation))
    for index in range(m):
        smaller = int(np.sum(permutation[index + 1 :] < permutation[index]))
        code = code * (m - index) + smaller
    return int(code)


def dense_amotf_single(
    series: np.ndarray,
    *,
    m: int,
    tau: int,
    spans: List[int],
    image_size: int,
    transition_mix: float,
    signed_mix: float,
    emd_mix: float,
) -> np.ndarray:
    values = np.asarray(series, dtype=np.float32)
    window_count = int(values.size) - (int(m) - 1) * int(tau)
    required_windows = int(image_size) + max(int(span) for span in spans)
    if window_count < required_windows:
        raise ValueError(f"series has {window_count} ordinal windows, expected at least {required_windows}")
    windows = np.stack(
        [values[index : index + int(m) * int(tau) : int(tau)] for index in range(window_count)],
        axis=0,
    )
    patterns = np.asarray(
        [permutation_to_index(np.argsort(window, kind="stable")) for window in windows],
        dtype=np.int32,
    )
    sorted_windows = np.sort(windows, axis=1, kind="stable")
    display_windows = windows[: int(image_size)]
    display_patterns = patterns[: int(image_size)]
    num_patterns = int(math.factorial(int(m)))

    pair_delta = display_windows[None, :, :] - display_windows[:, None, :]
    signed = np.mean(pair_delta, axis=2)
    signed_scale = max(float(np.max(np.abs(signed))), 1.0e-6)
    signed_unit = np.clip(0.5 + 0.5 * signed / signed_scale, 0.0, 1.0)
    display_sorted_windows = sorted_windows[: int(image_size)]
    emd_delta = display_sorted_windows[None, :, :] - display_sorted_windows[:, None, :]
    emd = np.mean(np.abs(emd_delta), axis=2)
    emd_scale = max(float(np.max(emd)), 1.0e-6)
    emd_unit = np.clip(emd / emd_scale, 0.0, 1.0)

    channels: List[np.ndarray] = []
    for span in spans:
        span = int(span)
        transition = np.zeros((num_patterns, num_patterns), dtype=np.float32)
        limit = int(len(patterns) - span)
        for index in range(max(0, limit)):
            source = int(patterns[index])
            target = int(patterns[index + span])
            weight = 1.0 + float(np.mean(np.abs(sorted_windows[index] - sorted_windows[index + span])))
            transition[source, target] += weight
        transition = transition / np.maximum(transition.sum(axis=1, keepdims=True), 1.0e-6)
        transition_field = transition[display_patterns[:, None], display_patterns[None, :]]
        transition_scale = max(float(np.max(transition_field)), 1.0e-6)
        transition_field = np.clip(transition_field / transition_scale, 0.0, 1.0)
        field = (
            float(transition_mix) * transition_field
            + float(signed_mix) * signed_unit
            + float(emd_mix) * emd_unit
        )
        channels.append(normalize_minmax(field))
    return np.stack(channels, axis=2).astype(np.float32)


def encode_phase(
    phase: pd.DataFrame,
    *,
    m: int,
    tau: int,
    spans: List[int],
    image_size: int,
    transition_mix: float,
    signed_mix: float,
    emd_mix: float,
) -> np.ndarray:
    required = int(image_size) + max(spans) + (int(m) - 1) * int(tau)
    voltage = normalize_minmax(resample_by_soc(phase["soc"].to_numpy(), phase["voltage"].to_numpy(), required))
    current = normalize_maxabs(resample_by_soc(phase["soc"].to_numpy(), phase["current"].to_numpy(), required))
    voltage_field = dense_amotf_single(
        voltage,
        m=m,
        tau=tau,
        spans=spans,
        image_size=image_size,
        transition_mix=transition_mix,
        signed_mix=signed_mix,
        emd_mix=emd_mix,
    )
    current_field = dense_amotf_single(
        current,
        m=m,
        tau=tau,
        spans=spans,
        image_size=image_size,
        transition_mix=transition_mix,
        signed_mix=signed_mix,
        emd_mix=emd_mix,
    )
    return np.concatenate([voltage_field, current_field], axis=2).astype(np.float32)


def valid_npz(path: Path, image_size: int, channels: int) -> bool:
    try:
        with np.load(path, allow_pickle=False) as saved:
            tensor = np.asarray(saved["amotf"])
        return (
            tensor.shape == (int(image_size), int(image_size), int(channels))
            and tensor.dtype == np.float32
            and bool(np.isfinite(tensor).all())
            and float(tensor.min()) >= -1.0e-6
            and float(tensor.max()) <= 1.0 + 1.0e-6
        )
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return False


def save_npz_atomic(path: Path, tensor: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, amotf=np.asarray(tensor, dtype=np.float32))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def encode_one_sample(task: Dict[str, object]) -> str:
    charge_npz = Path(str(task["charge_npz"]))
    discharge_npz = Path(str(task["discharge_npz"]))
    channels = 2 * len(task["spans"])
    if not bool(task["overwrite"]) and valid_npz(charge_npz, int(task["image_size"]), channels) and valid_npz(
        discharge_npz, int(task["image_size"]), channels
    ):
        return "exists"
    curve = read_curve_csv(Path(str(task["curve_csv"])))
    if curve is None:
        return "curve_read_failed"
    charge, discharge = split_charge_discharge(curve)
    if charge is None or discharge is None:
        return "split_failed"
    kwargs = {
        "m": int(task["m"]),
        "tau": int(task["tau"]),
        "spans": [int(value) for value in task["spans"]],
        "image_size": int(task["image_size"]),
        "transition_mix": float(task["transition_mix"]),
        "signed_mix": float(task["signed_mix"]),
        "emd_mix": float(task["emd_mix"]),
    }
    try:
        save_npz_atomic(charge_npz, encode_phase(charge, **kwargs))
        save_npz_atomic(discharge_npz, encode_phase(discharge, **kwargs))
    except Exception:
        return "encode_failed"
    return "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Cursor-locked dense AMOTF-v2 NPZ fields in place.")
    parser.add_argument("--labels_csv", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--encoding_dir", default="amotf_v2_dense_m5")
    parser.add_argument("--variant", default="dense_field_m5")
    parser.add_argument("--image_size", type=int, default=120)
    parser.add_argument("--m", type=int, default=5)
    parser.add_argument("--tau", type=int, default=1)
    parser.add_argument("--spans", default="1,2,4,8")
    parser.add_argument("--transition_mix", type=float, default=0.50)
    parser.add_argument("--signed_mix", type=float, default=0.25)
    parser.add_argument("--emd_mix", type=float, default=0.25)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Smoke-test only: process the first N labels.")
    parser.add_argument("--summary", default="")
    args = parser.parse_args()

    spans = [int(value) for value in str(args.spans).split(",") if str(value).strip()]
    if not spans or any(value <= 0 for value in spans):
        raise ValueError("--spans must contain positive integers")
    mix_sum = float(args.transition_mix) + float(args.signed_mix) + float(args.emd_mix)
    if abs(mix_sum - 1.0) > 1.0e-6:
        raise ValueError("transition_mix + signed_mix + emd_mix must equal 1")

    with Path(args.labels_csv).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if int(args.limit) > 0:
        rows = rows[: int(args.limit)]
    tasks: List[Dict[str, object]] = []
    for row in rows:
        source_dir = resolve_path(str(row.get("original_path", "")), args.data_root)
        encoding_dir = source_dir / str(args.encoding_dir).strip("/\\")
        tasks.append(
            {
                "curve_csv": str(source_dir / "curve.csv"),
                "charge_npz": str(encoding_dir / "amotf_charge.npz"),
                "discharge_npz": str(encoding_dir / "amotf_discharge.npz"),
                "m": args.m,
                "tau": args.tau,
                "spans": spans,
                "image_size": args.image_size,
                "transition_mix": args.transition_mix,
                "signed_mix": args.signed_mix,
                "emd_mix": args.emd_mix,
                "overwrite": args.overwrite,
            }
        )

    started = time.time()
    counts: Counter[str] = Counter()
    workers = max(1, int(args.num_workers))
    if workers == 1:
        iterator = map(encode_one_sample, tasks)
        pool = None
    else:
        pool = mp.get_context("spawn").Pool(processes=workers)
        iterator = pool.imap_unordered(encode_one_sample, tasks, chunksize=8)
    try:
        for completed, status in enumerate(iterator, start=1):
            counts[status] += 1
            if completed == 1 or completed % 100 == 0 or completed == len(tasks):
                elapsed = max(time.time() - started, 1.0e-6)
                print(
                    f"progress={completed}/{len(tasks)} rate={completed / elapsed:.2f}/s statuses={dict(counts)}",
                    flush=True,
                )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    summary_path = Path(args.summary) if args.summary else Path(args.labels_csv).with_name(
        f"{args.variant}_build_summary.json"
    )
    builder_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary = {
        "implementation": "server88_verified_dense_field_m5",
        "builder_sha256": builder_sha256,
        "variant": args.variant,
        "encoding_dir": args.encoding_dir,
        "image_size": args.image_size,
        "m": args.m,
        "tau": args.tau,
        "spans": spans,
        "mix": {
            "transition": args.transition_mix,
            "signed": args.signed_mix,
            "emd": args.emd_mix,
        },
        "total": len(tasks),
        "statuses": dict(counts),
        "complete": int(counts["ok"] + counts["exists"]) == len(tasks),
        "elapsed_seconds": time.time() - started,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    os.replace(temporary, summary_path)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
