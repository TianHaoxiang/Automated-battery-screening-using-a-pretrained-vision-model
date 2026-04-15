import json
import logging
import math
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


def setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def safe_filename(name: str) -> str:
    name = str(name)
    name = re.sub(r"\s+", "_", name.strip())
    name = re.sub(r"[^0-9A-Za-z_]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_") or "unknown"


def _normalize_col(col: str) -> str:
    col = str(col)
    col = re.sub(r"\s+", " ", col).strip().lower()
    return col


def normalize_id_for_match(value: str) -> str:
    value = str(value).strip().lower()
    # Replace any non-alphanumeric blocks with a single underscore
    value = re.sub(r"[^0-9a-z]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def _meta_get(meta: Dict[str, Any], candidates: List[str]) -> Any:
    if not meta:
        return None
    norm_map = {normalize_id_for_match(k): k for k in meta.keys()}
    for cand in candidates:
        nk = normalize_id_for_match(cand)
        k = norm_map.get(nk)
        if k is not None:
            return meta.get(k)
    return None


def extract_common_cell_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "nominal_capacity_Ah": _meta_get(meta, ["Ah (标称容量/标况容量)", "ah", "ah (nominal)"]),
        "anode": _meta_get(meta, ["anode", "an\node"]),
        "cathode": _meta_get(meta, ["cathode"]),
        "source": _meta_get(meta, ["source"]),
        "form_factor": _meta_get(meta, ["form_factor", "form factor"]),
        "tester": _meta_get(meta, ["tester"]),
        "temperature_C": _meta_get(meta, ["temperature", "temperature (c)"]),
        "soc_min": _meta_get(meta, ["soc_min", "soc min"]),
        "soc_max": _meta_get(meta, ["soc_max", "soc max"]),
        "crate_charge": _meta_get(meta, ["crate_c", "c-rate charge", "crate charge"]),
        "crate_discharge": _meta_get(meta, ["crate_d", "c-rate discharge", "crate discharge"]),
    }


def read_dataset_info_xlsx(xlsx_path: str, debug: bool = False) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Read Battery_Archive/dataset_info.xlsx.

    Note: in this local copy, the Excel is a *cell list* (one row per Cell ID) with meta columns.
    """

    df = pd.read_excel(xlsx_path)
    if debug:
        LOGGER.debug("dataset_info.xlsx shape=%s cols=%s", df.shape, list(df.columns))

    norm_cols = {_normalize_col(c): c for c in df.columns}

    cell_col = None
    for cand in ["cell id", "cell_id", "cellid"]:
        if cand in norm_cols:
            cell_col = norm_cols[cand]
            break
    if cell_col is None:
        raise ValueError(f"Cannot find Cell ID column in {xlsx_path}. Columns={list(df.columns)}")

    df = df.copy()
    df[cell_col] = df[cell_col].astype(str)

    meta_by_cell: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        cell_id = str(row[cell_col])
        meta: Dict[str, Any] = {}
        for c in df.columns:
            v = row[c]
            if pd.isna(v):
                meta[str(c)] = None
            elif isinstance(v, (np.integer, np.floating)):
                if isinstance(v, np.floating) and math.isnan(float(v)):
                    meta[str(c)] = None
                else:
                    meta[str(c)] = float(v) if isinstance(v, np.floating) else int(v)
            else:
                meta[str(c)] = v
        # Store both exact key and a normalized key for robust matching.
        meta_by_cell[cell_id] = meta
        norm_key = normalize_id_for_match(cell_id)
        if norm_key and norm_key not in meta_by_cell:
            meta_by_cell[norm_key] = meta
        elif norm_key and norm_key in meta_by_cell and debug:
            LOGGER.debug("Duplicate normalized Cell ID key in dataset_info.xlsx: %s", norm_key)

    return df, meta_by_cell


@dataclass(frozen=True)
class CellFiles:
    cell_id: str
    timeseries_csv: str
    cycle_csv: Optional[str]


def discover_flat_cell_files(dataset_dir: str) -> List[CellFiles]:
    """Discover cells in datasets like CALCE/HNEI/SNL*/UL-Purdue where files are flat pairs.

    Expected naming:
    - <cell_id>_timeseries.csv or <cell_id>_timeseries_data.csv
    - <cell_id>_cycle_data.csv

    Returns CellFiles for any prefix that has a timeseries file.
    """

    suffix_ts = ("_timeseries.csv", "_timeseries_data.csv")
    suffix_cycle = "_cycle_data.csv"

    files = [f for f in os.listdir(dataset_dir) if f.lower().endswith(".csv")]
    ts_by_prefix: Dict[str, str] = {}
    cycle_by_prefix: Dict[str, str] = {}

    for fn in files:
        lower = fn.lower()
        full = os.path.join(dataset_dir, fn)
        for sfx in suffix_ts:
            if lower.endswith(sfx):
                prefix = fn[: -len(sfx)]
                ts_by_prefix[prefix] = full
                break
        if lower.endswith(suffix_cycle):
            prefix = fn[: -len(suffix_cycle)]
            cycle_by_prefix[prefix] = full

    prefixes = sorted(ts_by_prefix.keys())
    out: List[CellFiles] = []
    for p in prefixes:
        out.append(CellFiles(cell_id=p, timeseries_csv=ts_by_prefix[p], cycle_csv=cycle_by_prefix.get(p)))
    return out


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_timeseries_csv(path: str) -> pd.DataFrame:
    # Reduce memory: read only needed columns if present.
    header = pd.read_csv(path, nrows=0)
    wanted_norm = {
        "date_time",
        "datetime",
        "date time",
        "test_time (s)",
        "test_time(s)",
        "test time (s)",
        "test time(s)",
        "cycle_index",
        "cycle index",
        "current (a)",
        "current",
        "voltage (v)",
        "voltage",
        "charge_capacity (ah)",
        "charge capacity (ah)",
        "charge_capacity",
        "discharge_capacity (ah)",
        "discharge capacity (ah)",
        "discharge_capacity",
    }

    usecols = [c for c in header.columns if _normalize_col(c) in wanted_norm]
    df = pd.read_csv(path, low_memory=False, usecols=usecols if usecols else None)

    # Standardize common columns
    rename_map: Dict[str, str] = {}
    for c in df.columns:
        nc = _normalize_col(c)
        if nc in {"date_time", "datetime", "date time"}:
            rename_map[c] = "datetime"
        elif nc in {"test_time (s)", "test_time(s)", "test time (s)", "test time(s)"}:
            rename_map[c] = "t_s"
        elif nc in {"cycle_index", "cycle index"}:
            rename_map[c] = "cycle_index"
        elif nc in {"current (a)", "current"}:
            rename_map[c] = "current_A"
        elif nc in {"voltage (v)", "voltage"}:
            rename_map[c] = "voltage_V"
        elif nc in {"charge_capacity (ah)", "charge capacity (ah)", "charge_capacity"}:
            rename_map[c] = "charge_capacity_Ah"
        elif nc in {"discharge_capacity (ah)", "discharge capacity (ah)", "discharge_capacity"}:
            rename_map[c] = "discharge_capacity_Ah"

    df = df.rename(columns=rename_map)

    required = ["cycle_index", "current_A", "voltage_V"]
    for r in required:
        if r not in df.columns:
            raise ValueError(f"Missing required column '{r}' in {path}. Columns={list(df.columns)}")

    df["cycle_index"] = _coerce_numeric(df["cycle_index"]).round().astype("Int64")
    df["current_A"] = _coerce_numeric(df["current_A"])
    df["voltage_V"] = _coerce_numeric(df["voltage_V"])

    if "t_s" in df.columns:
        df["t_s"] = _coerce_numeric(df["t_s"])

    if "datetime" in df.columns:
        # Keep raw string, but try parse when needed
        df["datetime"] = df["datetime"].astype(str)

    df = df.dropna(subset=["cycle_index", "current_A", "voltage_V"]).copy()
    return df


def load_cycle_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    rename_map: Dict[str, str] = {}
    for c in df.columns:
        nc = _normalize_col(c)
        if nc in {"cycle_index", "cycle index"}:
            rename_map[c] = "cycle_index"
        elif nc in {"min_voltage (v)", "min voltage (v)", "min_voltage"}:
            rename_map[c] = "min_voltage_V"
        elif nc in {"max_voltage (v)", "max voltage (v)", "max_voltage"}:
            rename_map[c] = "max_voltage_V"

    df = df.rename(columns=rename_map)

    if "cycle_index" in df.columns:
        df["cycle_index"] = _coerce_numeric(df["cycle_index"]).round().astype("Int64")

    if "min_voltage_V" in df.columns:
        df["min_voltage_V"] = _coerce_numeric(df["min_voltage_V"])
    if "max_voltage_V" in df.columns:
        df["max_voltage_V"] = _coerce_numeric(df["max_voltage_V"])

    return df


def infer_phases_by_current_and_voltage(
    df_cycle: pd.DataFrame,
    min_points: int,
    eps_current: float = 1e-6,

) -> Tuple[pd.DataFrame, pd.DataFrame, bool, str, str]:
    """Infer charge/discharge segments for a cycle.

    Returns (df_charge, df_discharge, phase_inferred, reason, method)

    method in:
    - capacity_delta
    - current_sign
    """

    if "current_A" not in df_cycle.columns or "voltage_V" not in df_cycle.columns:
        return pd.DataFrame(), pd.DataFrame(), True, "missing_current_or_voltage", "current_sign"

    # Prefer capacity counters if available (more robust than current sign convention)
    if "charge_capacity_Ah" in df_cycle.columns and "discharge_capacity_Ah" in df_cycle.columns:
        df_sorted = df_cycle.sort_values("t_s") if "t_s" in df_cycle.columns else df_cycle
        ch_cap = pd.to_numeric(df_sorted["charge_capacity_Ah"], errors="coerce").astype(float)
        dch_cap = pd.to_numeric(df_sorted["discharge_capacity_Ah"], errors="coerce").astype(float)

        d_ch = ch_cap.diff().fillna(0.0)
        d_dch = dch_cap.diff().fillna(0.0)

        # Threshold small negative noise
        charge_mask = d_ch > 1e-8
        discharge_mask = d_dch > 1e-8

        df_charge = df_sorted.loc[charge_mask].copy()
        df_discharge = df_sorted.loc[discharge_mask].copy()

        if len(df_charge) >= min_points and len(df_discharge) >= min_points:
            return df_charge, df_discharge, True, "phase_inferred_by_capacity_delta", "capacity_delta"

    current = df_cycle["current_A"]

    charge_mask = current > eps_current
    discharge_mask = current < -eps_current

    df_charge = df_cycle.loc[charge_mask].copy()
    df_discharge = df_cycle.loc[discharge_mask].copy()

    # If one side missing, try reversed polarity
    if len(df_charge) < min_points or len(df_discharge) < min_points:
        charge_mask2 = current < -eps_current
        discharge_mask2 = current > eps_current
        df_charge2 = df_cycle.loc[charge_mask2].copy()
        df_discharge2 = df_cycle.loc[discharge_mask2].copy()

        # Choose the split that produces both sides
        if len(df_charge2) >= min_points and len(df_discharge2) >= min_points:
            df_charge, df_discharge = df_charge2, df_discharge2

    if len(df_charge) < min_points or len(df_discharge) < min_points:
        return pd.DataFrame(), pd.DataFrame(), True, "insufficient_charge_or_discharge_points", "current_sign"

    # Validate by voltage trend: charge should increase, discharge should decrease.
    def _trend_ok(df_seg: pd.DataFrame, expect_increasing: bool) -> bool:
        if "t_s" in df_seg.columns and df_seg["t_s"].notna().any():
            df_seg = df_seg.sort_values("t_s")
        v0 = float(df_seg["voltage_V"].iloc[0])
        v1 = float(df_seg["voltage_V"].iloc[-1])
        return (v1 >= v0) if expect_increasing else (v1 <= v0)

    charge_ok = _trend_ok(df_charge, expect_increasing=True)
    discharge_ok = _trend_ok(df_discharge, expect_increasing=False)

    if not (charge_ok and discharge_ok):
        # Swap and re-check
        df_charge_s, df_discharge_s = df_discharge.copy(), df_charge.copy()
        if _trend_ok(df_charge_s, True) and _trend_ok(df_discharge_s, False):
            df_charge, df_discharge = df_charge_s, df_discharge_s
            return df_charge, df_discharge, True, "phase_swapped_by_voltage_trend", "current_sign"

    return df_charge, df_discharge, True, "phase_inferred_by_current_sign", "current_sign"


def is_complete_cycle(
    df_cycle: pd.DataFrame,
    cycle_stats: Optional[Dict[str, float]],
    min_points: int,
    v_tol: float = 0.02,
    dv_min: float = 0.05,
) -> Tuple[bool, str, Optional[pd.DataFrame], Optional[pd.DataFrame], bool, str]:
    """Check if a cycle contains a complete charge + discharge.

    Returns:
    - is_complete
    - reason
    - df_charge
    - df_discharge
    - phase_inferred
    """

    needed = ["current_A", "voltage_V"]
    for c in needed:
        if c not in df_cycle.columns:
            return False, f"missing_column:{c}", None, None, True, "unknown"

    if "t_s" not in df_cycle.columns or df_cycle["t_s"].isna().all():
        # SOC requires time; we treat missing time as incomplete.
        return False, "missing_time_column", None, None, True, "unknown"

    df_charge, df_discharge, phase_inferred, phase_reason, phase_method = infer_phases_by_current_and_voltage(
        df_cycle, min_points=min_points
    )
    if df_charge.empty or df_discharge.empty:
        return False, phase_reason, None, None, phase_inferred, phase_method

    df_charge = df_charge.sort_values("t_s")
    df_discharge = df_discharge.sort_values("t_s")

    def _endpoint_dv(df_seg: pd.DataFrame) -> float:
        k = min(5, len(df_seg))
        v_start = float(df_seg["voltage_V"].iloc[:k].median())
        v_end = float(df_seg["voltage_V"].iloc[-k:].median())
        return v_end - v_start

    dv_charge = _endpoint_dv(df_charge)
    dv_discharge = _endpoint_dv(df_discharge)

    if dv_charge < dv_min:
        return False, f"charge_voltage_not_rising(dv={dv_charge:.3f})", None, None, phase_inferred, phase_method
    if dv_discharge > -dv_min:
        return False, f"discharge_voltage_not_falling(dv={dv_discharge:.3f})", None, None, phase_inferred, phase_method

    # Cutoff validation (if we have expected min/max for this cycle).
    v_max_expected = None
    v_min_expected = None
    if cycle_stats is not None:
        v_max_expected = cycle_stats.get("max_voltage_V")
        v_min_expected = cycle_stats.get("min_voltage_V")

    if v_max_expected is None:
        v_max_expected = float(df_cycle["voltage_V"].quantile(0.995))
    if v_min_expected is None:
        v_min_expected = float(df_cycle["voltage_V"].quantile(0.005))

    v_charge_max = float(df_charge["voltage_V"].max())
    v_discharge_min = float(df_discharge["voltage_V"].min())

    if v_charge_max < v_max_expected - v_tol:
        return (
            False,
            f"charge_not_reach_upper_cutoff(vmax={v_charge_max:.3f}, expected>={v_max_expected - v_tol:.3f})",
            None,
            None,
            phase_inferred,
            phase_method,
        )

    if v_discharge_min > v_min_expected + v_tol:
        return (
            False,
            f"discharge_not_reach_lower_cutoff(vmin={v_discharge_min:.3f}, expected<={v_min_expected + v_tol:.3f})",
            None,
            None,
            phase_inferred,
            phase_method,
        )

    return True, "ok", df_charge, df_discharge, phase_inferred, phase_method


def compute_soc_coulomb_counting(df_phase: pd.DataFrame, phase: str) -> pd.Series:
    """Compute SOC(0-100) for one phase by Coulomb counting.

    SOC 计算规则（强约束）：库仑计量

    给定时间序列 t (s) 与电流 I (A)，增量电量：
        dQ = I * dt / 3600  (Ah)

    本项目要求图的纵轴电流使用绝对值，且不同数据集电流正负可能不一致。
    因此这里对单段 phase 使用 |I| 来做相对 SOC：

    - Charge：累计电量从 0 归一化到 100%
    - Discharge：累计电量从 0 归一化到 100%，再映射为 100% -> 0%

    注意：这是“相对 SOC”（每个 phase 内部归一化），不使用历史信息。
    """

    if "t_s" not in df_phase.columns:
        raise ValueError("Missing t_s for SOC computation")

    t = pd.to_numeric(df_phase["t_s"], errors="coerce").astype(float)
    i_abs = pd.to_numeric(df_phase["current_A"], errors="coerce").abs().astype(float)

    dt = t.diff()
    # For safety: replace non-positive dt
    dt_pos = dt[dt > 0]
    fallback_dt = float(dt_pos.median()) if len(dt_pos) else 1.0
    dt = dt.fillna(0.0)
    dt = dt.where(dt > 0, fallback_dt)

    dQ = i_abs * dt / 3600.0
    cumQ = dQ.cumsum()

    total = float(cumQ.iloc[-1]) if len(cumQ) else 0.0
    if total <= 0:
        soc_rel = pd.Series(np.zeros(len(df_phase)), index=df_phase.index, dtype=float)
    else:
        soc_rel = 100.0 * (cumQ / total)

    if phase == "charge":
        return soc_rel
    if phase == "discharge":
        return 100.0 - soc_rel

    raise ValueError(f"Unknown phase: {phase}")


def _robust_dt_seconds(t_s: pd.Series) -> pd.Series:
    t = pd.to_numeric(t_s, errors="coerce").astype(float)
    dt = t.diff()
    dt_pos = dt[dt > 0]
    fallback_dt = float(dt_pos.median()) if len(dt_pos) else 1.0
    dt = dt.fillna(0.0)
    dt = dt.where(dt > 0, fallback_dt)
    return dt


def _integrated_charge_ah(df: pd.DataFrame, current_col: str) -> float:
    if df.empty:
        return 0.0
    if "t_s" not in df.columns:
        return 0.0
    dt = _robust_dt_seconds(df["t_s"])
    i = pd.to_numeric(df[current_col], errors="coerce").abs().astype(float)
    dQ = i * dt / 3600.0
    return float(dQ.sum(skipna=True))


def clean_cccv_transient(
    df_charge: pd.DataFrame,
    cutoff_max_V: Optional[float],
    *,
    v_switch_frac: float = 0.999,
    v_post_switch_min_frac: float = 0.99,
    spike_ratio: float = 1.05,
    zero_ratio: float = 0.2,
    mad_k: float = 6.0,
    transient_window_points: int = 12,
    max_remove_frac: float = 0.02,
    rolling_window: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean CC→CV switching transient in CCCV charge.

    Design goals:
    - Detect CC→CV switch by voltage reaching near cutoff.
    - Remove/repair short switching transient (current drop to ~0, spike, or voltage dip).
    - Enforce CCCV current shape:
        CC ~ constant
        CV monotone non-increasing and <= CC current
    - Keep charge conservation: relative error of sum(|I|*dt) should be small (reported).
    """

    info: Dict[str, Any] = {
        "enabled": False,
        "method": None,
        "num_points_modified": 0,
        "num_points_removed": 0,
        "energy_error_pct": None,
    }

    if df_charge is None or df_charge.empty:
        return df_charge, info

    needed = {"t_s", "voltage_V", "current_A"}
    if not needed.issubset(set(df_charge.columns)):
        return df_charge, info

    df = df_charge.copy()
    df = df.sort_values("t_s").reset_index(drop=True)

    v = pd.to_numeric(df["voltage_V"], errors="coerce").astype(float)
    i_abs = pd.to_numeric(df["current_A"], errors="coerce").abs().astype(float)

    if cutoff_max_V is None or (isinstance(cutoff_max_V, float) and not np.isfinite(cutoff_max_V)):
        cutoff = float(v.quantile(0.995))
    else:
        cutoff = float(cutoff_max_V)
    if not np.isfinite(cutoff) or cutoff <= 0:
        return df_charge, info

    v_switch = cutoff * v_switch_frac
    switch_pos_arr = np.where((v.to_numpy() >= v_switch) & np.isfinite(v.to_numpy()))[0]
    if len(switch_pos_arr) == 0:
        return df_charge, info
    switch_pos = int(switch_pos_arr[0])

    # Estimate CC current from points before switch (exclude last window to avoid transient)
    cc_end = max(0, switch_pos - transient_window_points)
    cc_region = i_abs.iloc[: max(1, cc_end)].replace([np.inf, -np.inf], np.nan).dropna()
    if len(cc_region) < 10:
        cc_region = i_abs.replace([np.inf, -np.inf], np.nan).dropna()
    if len(cc_region) == 0:
        return df_charge, info
    i_cc = float(np.nanmedian(cc_region.to_numpy()))
    if not np.isfinite(i_cc) or i_cc <= 0:
        return df_charge, info

    n0 = int(len(df))
    q_before = _integrated_charge_ah(df.assign(_i=i_abs), current_col="_i")

    # Identify transient candidates around switch
    lo = max(0, switch_pos - transient_window_points)
    hi = min(n0 - 1, switch_pos + transient_window_points)
    idx_window = np.arange(lo, hi + 1)

    v_arr = v.to_numpy()
    i_arr = i_abs.to_numpy()

    transient_mask = np.zeros(n0, dtype=bool)
    transient_mask[idx_window] |= (i_arr[idx_window] < zero_ratio * i_cc)
    transient_mask[idx_window] |= (i_arr[idx_window] > spike_ratio * i_cc)
    # After switch, voltage should not dip far below cutoff; if it does, treat as switch transient
    post_switch = idx_window[idx_window > switch_pos]
    if len(post_switch):
        transient_mask[post_switch] |= (v_arr[post_switch] < cutoff * v_post_switch_min_frac)

    # Conservative removal for points with voltage dip or near-zero current
    remove_mask = transient_mask & ((v_arr < cutoff * v_post_switch_min_frac) | (i_arr < zero_ratio * i_cc))
    num_remove = int(remove_mask.sum())
    if num_remove > int(math.ceil(max_remove_frac * n0)):
        # Too many points would be removed; keep them and only clip/repair current.
        remove_mask[:] = False
        num_remove = 0

    if num_remove:
        df = df.loc[~remove_mask].reset_index(drop=True)
        v = pd.to_numeric(df["voltage_V"], errors="coerce").astype(float)
        i_abs = pd.to_numeric(df["current_A"], errors="coerce").abs().astype(float)

        n1 = int(len(df))
        v_arr = v.to_numpy()
        i_arr = i_abs.to_numpy()
        switch_pos_arr = np.where((v_arr >= v_switch) & np.isfinite(v_arr))[0]
        switch_pos = int(switch_pos_arr[0]) if len(switch_pos_arr) else max(0, n1 - 1)
    else:
        n1 = n0

    # Split CC/CV by first near-cutoff sample in time
    switch_pos = int(min(max(switch_pos, 0), n1 - 1))
    cc_idx = np.arange(0, switch_pos + 1)
    cv_idx = np.arange(switch_pos + 1, n1)

    # CC: robust outlier repair using MAD (and hard bounds)
    i_abs_clean = i_abs.copy()
    if len(cc_idx) >= 5:
        cc_vals = i_abs_clean.iloc[cc_idx]
        med = float(np.nanmedian(cc_vals.to_numpy()))
        mad = float(np.nanmedian(np.abs(cc_vals.to_numpy() - med)))
        mad = mad if (np.isfinite(mad) and mad > 0) else 0.0

        if mad > 0:
            outlier = np.abs(i_abs_clean.iloc[cc_idx] - med) > (mad_k * mad)
        else:
            # When CC is nearly perfectly constant, MAD can become 0.
            # Fall back to a small relative tolerance to catch switch glitches.
            outlier = np.abs(i_abs_clean.iloc[cc_idx] - med) > (0.05 * med)

        hard_bad = (i_abs_clean.iloc[cc_idx] < zero_ratio * i_cc) | (i_abs_clean.iloc[cc_idx] > 1.5 * i_cc)
        bad_mask_cc = outlier | hard_bad
        if bad_mask_cc.any():
            i_abs_clean.loc[bad_mask_cc.index[bad_mask_cc]] = i_cc

    # CV: clip to <= CC and enforce monotone non-increasing
    if len(cv_idx) >= 3:
        cv_vals = i_abs_clean.iloc[cv_idx].copy()
        cv_vals = cv_vals.clip(lower=0.0, upper=i_cc)

        if rolling_window and rolling_window >= 3 and len(cv_vals) >= rolling_window:
            cv_vals_sm = cv_vals.rolling(window=int(rolling_window), center=True, min_periods=1).median()
        else:
            cv_vals_sm = cv_vals

        cv_np = cv_vals_sm.to_numpy(dtype=float)
        # Running minimum enforces monotone non-increasing
        cv_mono = np.minimum.accumulate(cv_np)
        i_abs_clean.iloc[cv_idx] = cv_mono

    # Ensure CV start does not exceed CC
    if len(cv_idx) and np.isfinite(i_abs_clean.iloc[cv_idx[0]]):
        i_abs_clean.iloc[cv_idx[0]] = min(float(i_abs_clean.iloc[cv_idx[0]]), i_cc)

    # Clip any numerical negatives
    i_abs_clean = i_abs_clean.clip(lower=0.0)

    # Energy conservation stats + optional correction
    q_after = _integrated_charge_ah(df.assign(_i=i_abs_clean), current_col="_i")
    energy_error_pct = (100.0 * float((q_after - q_before) / q_before)) if q_before > 0 else None

    if q_before > 0 and q_after > 0:
        rel_err = float((q_after - q_before) / q_before)
        if abs(rel_err) > 0.01:
            scale = float(q_before / q_after)
            i_abs_clean = (i_abs_clean * scale).astype(float)

            # Re-enforce CCCV constraints after scaling
            i_cc2 = float(np.nanmedian(i_abs_clean.iloc[cc_idx].to_numpy())) if len(cc_idx) else i_cc
            if not np.isfinite(i_cc2) or i_cc2 <= 0:
                i_cc2 = i_cc

            if len(cv_idx) >= 1:
                cv_vals2 = i_abs_clean.iloc[cv_idx].clip(lower=0.0, upper=i_cc2)
                cv_mono2 = np.minimum.accumulate(cv_vals2.to_numpy(dtype=float))
                i_abs_clean.iloc[cv_idx] = cv_mono2

            i_abs_clean = i_abs_clean.clip(lower=0.0)

            q_after = _integrated_charge_ah(df.assign(_i=i_abs_clean), current_col="_i")
            energy_error_pct = 100.0 * float((q_after - q_before) / q_before)

    # Track modifications (after final correction)
    changed = (
        pd.to_numeric(i_abs, errors="coerce").astype(float)
        - pd.to_numeric(i_abs_clean, errors="coerce").astype(float)
    ).abs()
    num_modified = int((changed > 1e-12).sum())

    # Write back cleaned current (keep as positive magnitude; downstream always uses abs anyway)
    df["current_A"] = i_abs_clean

    info.update(
        {
            "enabled": True,
            "method": "voltage_threshold+transient_remove+cc_mad_repair+cv_monotone_clip",
            "num_points_modified": int(num_modified),
            "num_points_removed": int(num_remove),
            "energy_error_pct": energy_error_pct,
            "cutoff_max_V_used": cutoff,
            "i_cc_est_A": i_cc,
            "switch_voltage_thr_V": v_switch,
        }
    )

    return df, info


def build_sample_curve(
    df_charge: pd.DataFrame,
    df_discharge: pd.DataFrame,
    *,
    dataset_name_raw: Optional[str] = None,
    cutoff_max_V: Optional[float] = None,
    cccv_info_out: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    df_charge = df_charge.copy()
    df_discharge = df_discharge.copy()

    cccv_info: Dict[str, Any] = {
        "enabled": False,
        "method": None,
        "num_points_modified": 0,
        "num_points_removed": 0,
        "energy_error_pct": None,
    }

    # Only clean CALCE charge transient (CC→CV) by default.
    if dataset_name_raw == "CALCE":
        df_charge, cccv_info = clean_cccv_transient(df_charge, cutoff_max_V=cutoff_max_V)

    if cccv_info_out is not None:
        cccv_info_out.update(cccv_info)

    df_charge["phase"] = "charge"
    df_discharge["phase"] = "discharge"

    df_charge["soc_pct"] = compute_soc_coulomb_counting(df_charge, phase="charge")
    df_discharge["soc_pct"] = compute_soc_coulomb_counting(df_discharge, phase="discharge")

    # Model-facing curve: do NOT include cycle index or any history fields.
    keep_cols = ["soc_pct", "voltage_V", "current_A", "current_abs_A", "phase", "t_s"]
    if "datetime" in df_charge.columns or "datetime" in df_discharge.columns:
        keep_cols.append("datetime")

    out = pd.concat([df_charge, df_discharge], axis=0, ignore_index=True)
    out["current_A"] = pd.to_numeric(out["current_A"], errors="coerce").abs()
    out["current_abs_A"] = out["current_A"]

    # Keep only known columns if they exist
    keep_cols_existing = [c for c in keep_cols if c in out.columns]
    out = out[keep_cols_existing]

    return out


def plot_charge_discharge(
    curve: pd.DataFrame,
    out_charge_png: str,
    out_discharge_png: str,
    title_prefix: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _safe_upper_limit(values: pd.Series) -> float:
        v = pd.to_numeric(values, errors="coerce").astype(float)
        if v.empty:
            return 1.0
        vmax = float(np.nanmax(np.abs(v.to_numpy()))) if np.isfinite(v).any() else 1.0
        if not np.isfinite(vmax) or vmax <= 0:
            return 1.0
        return 1.2 * vmax

    def _plot_one(df_phase: pd.DataFrame, phase: str, out_png: str) -> None:
        df_phase = df_phase.sort_values("soc_pct") if phase == "charge" else df_phase.sort_values("soc_pct", ascending=False)

        fig, ax_v = plt.subplots(figsize=(10, 4))
        ax_i = ax_v.twinx()

        ax_v.plot(df_phase["soc_pct"], df_phase["voltage_V"], color="tab:blue", label="Voltage (V)")
        ax_i.plot(df_phase["soc_pct"], df_phase["current_A"], color="tab:orange", label="Current |A|")

        ax_v.set_xlabel("SOC (%)")
        ax_v.set_ylabel("Voltage (V)")
        ax_i.set_ylabel("Current (A, abs)")

        # Y axis ranges: 0 .. 1.2 * max(abs(value)) for each image
        try:
            ax_v.set_ylim(0.0, _safe_upper_limit(df_phase["voltage_V"]))
            ax_i.set_ylim(0.0, _safe_upper_limit(df_phase["current_A"]))
        except Exception:
            LOGGER.debug("Failed to set y-limits for plot (%s)", phase, exc_info=True)

        if phase == "charge":
            ax_v.set_xlim(0, 100)
            title = f"{title_prefix} | Charge"
        else:
            ax_v.set_xlim(100, 0)
            title = f"{title_prefix} | Discharge"

        ax_v.set_title(title)

        # Combine legends
        lines1, labels1 = ax_v.get_legend_handles_labels()
        lines2, labels2 = ax_i.get_legend_handles_labels()
        ax_v.legend(lines1 + lines2, labels1 + labels2, loc="best")

        fig.tight_layout()
        fig.savefig(out_png, dpi=200)
        plt.close(fig)

    df_c = curve[curve["phase"] == "charge"].copy()
    df_d = curve[curve["phase"] == "discharge"].copy()

    _plot_one(df_c, "charge", out_charge_png)
    _plot_one(df_d, "discharge", out_discharge_png)


def _get_cycle_stats(cycle_df: Optional[pd.DataFrame], cycle_index: int) -> Optional[Dict[str, float]]:
    if cycle_df is None or cycle_df.empty:
        return None
    if "cycle_index" not in cycle_df.columns:
        return None

    row = cycle_df.loc[cycle_df["cycle_index"] == cycle_index]
    if row.empty:
        return None

    r0 = row.iloc[0]
    out: Dict[str, float] = {}
    if "min_voltage_V" in row.columns and pd.notna(r0.get("min_voltage_V")):
        out["min_voltage_V"] = float(r0["min_voltage_V"])
    if "max_voltage_V" in row.columns and pd.notna(r0.get("max_voltage_V")):
        out["max_voltage_V"] = float(r0["max_voltage_V"])
    return out or None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def extract_samples_for_cell(
    dataset_name_raw: str,
    dataset_name_safe: str,
    cell_files: CellFiles,
    meta_by_cell: Dict[str, Dict[str, Any]],
    out_dataset_dir: str,
    mode: str,
    min_points: int,
    overwrite: bool,
    debug: bool,
) -> List[Dict[str, Any]]:
    """Extract samples for one cell (last_only or all_cycles).

    Returns index rows.
    """

    index_rows: List[Dict[str, Any]] = []

    try:
        ts = load_timeseries_csv(cell_files.timeseries_csv)
    except Exception as e:
        LOGGER.exception("Failed to read timeseries for %s: %s", cell_files.cell_id, e)
        index_rows.append(
            {
                "dataset_name": dataset_name_raw,
                "dataset_name_safe": dataset_name_safe,
                "cell_id": cell_files.cell_id,
                "mode": mode,
                "status": "failed",
                "fail_reason": f"read_timeseries_error:{type(e).__name__}:{e}",
            }
        )
        return index_rows

    cycle_df = None
    if cell_files.cycle_csv and os.path.exists(cell_files.cycle_csv):
        try:
            cycle_df = load_cycle_csv(cell_files.cycle_csv)
        except Exception as e:
            LOGGER.warning("Failed to read cycle_data for %s: %s", cell_files.cell_id, e)
            cycle_df = None

    cycles = sorted(ts["cycle_index"].dropna().astype(int).unique().tolist())
    if not cycles:
        index_rows.append(
            {
                "dataset_name": dataset_name_raw,
                "dataset_name_safe": dataset_name_safe,
                "cell_id": cell_files.cell_id,
                "mode": mode,
                "status": "failed",
                "fail_reason": "no_cycles_found",
            }
        )
        return index_rows

    cell_meta = meta_by_cell.get(cell_files.cell_id) or meta_by_cell.get(normalize_id_for_match(cell_files.cell_id)) or {}
    if not cell_meta and debug:
        LOGGER.debug("No meta found in dataset_info.xlsx for cell_id=%s", cell_files.cell_id)

    forbidden_features = [
        "cycle_index",
        "num_cycles",
        "cumulative_cycles",
        "capacity_fade_history",
        "soh_history",
    ]

    common_cell_meta = extract_common_cell_meta(cell_meta)

    def _export_one(cycle_index: int, fallback_steps: int) -> None:
        df_cycle = ts.loc[ts["cycle_index"].astype(int) == int(cycle_index)].copy()

        cycle_stats = _get_cycle_stats(cycle_df, int(cycle_index))
        ok, reason, df_charge, df_discharge, phase_inferred2, phase_method = is_complete_cycle(
            df_cycle,
            cycle_stats=cycle_stats,
            min_points=min_points,
        )

        if not ok:
            index_rows.append(
                {
                    "dataset_name": dataset_name_raw,
                    "dataset_name_safe": dataset_name_safe,
                    "cell_id": cell_files.cell_id,
                    "cycle_index": int(cycle_index),
                    "mode": mode,
                    "status": "skipped",
                    "is_complete": False,
                    "completeness_reason": reason,
                    "phase_inferred": bool(phase_inferred2),
                    "phase_inferred_method": phase_method,
                }
            )
            return

        assert df_charge is not None and df_discharge is not None

        cccv_info: Dict[str, Any] = {
            "enabled": False,
            "method": None,
            "num_points_modified": 0,
            "num_points_removed": 0,
            "energy_error_pct": None,
        }

        cutoff_for_clean = None
        if cycle_stats is not None:
            cutoff_for_clean = cycle_stats.get("max_voltage_V")

        curve = build_sample_curve(
            df_charge=df_charge,
            df_discharge=df_discharge,
            dataset_name_raw=dataset_name_raw,
            cutoff_max_V=cutoff_for_clean,
            cccv_info_out=cccv_info,
        )

        inferred_cutoff = {
            "min_voltage_V": float(df_cycle["voltage_V"].quantile(0.005)),
            "max_voltage_V": float(df_cycle["voltage_V"].quantile(0.995)),
            "note": "inferred from cycle quantiles",
        }

        sample_id = f"{safe_filename(cell_files.cell_id)}__cyc{int(cycle_index)}__{mode}"
        sample_dir = os.path.join(out_dataset_dir, "samples", sample_id)
        ensure_dir(sample_dir)

        curve_path = os.path.join(sample_dir, "curve.csv")
        meta_path = os.path.join(sample_dir, "meta.json")

        charge_png = os.path.join(
            sample_dir,
            f"charge_{dataset_name_safe}_{safe_filename(cell_files.cell_id)}_cyc{int(cycle_index)}_{mode}.png",
        )
        discharge_png = os.path.join(
            sample_dir,
            f"discharge_{dataset_name_safe}_{safe_filename(cell_files.cell_id)}_cyc{int(cycle_index)}_{mode}.png",
        )

        charge_png_std = os.path.join(sample_dir, "charge.png")
        discharge_png_std = os.path.join(sample_dir, "discharge.png")

        if (not overwrite) and (os.path.exists(curve_path) or os.path.exists(meta_path)):
            index_rows.append(
                {
                    "dataset_name": dataset_name_raw,
                    "dataset_name_safe": dataset_name_safe,
                    "cell_id": cell_files.cell_id,
                    "cycle_index": int(cycle_index),
                    "mode": mode,
                    "sample_id": sample_id,
                    "sample_dir": sample_dir,
                    "status": "exists",
                    "is_complete": True,
                    "completeness_reason": "ok",
                }
            )
            return

        curve.to_csv(curve_path, index=False)

        meta: Dict[str, Any] = {
            "dataset_name": dataset_name_raw,
            "dataset_name_safe": dataset_name_safe,
            "dataset_dir": os.path.dirname(cell_files.timeseries_csv),
            "cell_id": cell_files.cell_id,
            "cycle_index": int(cycle_index),
            "mode": mode,
            "cell_meta": common_cell_meta,
            "phase_inferred": bool(phase_inferred2),
            "phase_inferred_method": phase_method,
            "completeness": {
                "is_complete": True,
                "reason": "ok",
                "min_points": int(min_points),
                "fallback_steps": int(fallback_steps),
            },
            "cutoff_voltage": cycle_stats or inferred_cutoff,
            "cccv_transient_handling": cccv_info,
            "forbidden_features": forbidden_features,
            "source_meta_from_dataset_info": cell_meta or None,
            "missing_meta_reason": None if cell_meta else "cell_id_not_found_in_dataset_info.xlsx",
        }

        write_json(meta_path, meta)

        title_prefix = f"{dataset_name_raw} | {cell_files.cell_id} | cyc {int(cycle_index)} | {mode}"
        plot_charge_discharge(curve, out_charge_png=charge_png, out_discharge_png=discharge_png, title_prefix=title_prefix)

        # Also write standard names for convenience
        try:
            shutil.copyfile(charge_png, charge_png_std)
            shutil.copyfile(discharge_png, discharge_png_std)
        except Exception:
            LOGGER.debug("Failed to write standard PNG copies for %s", sample_id, exc_info=True)

        index_rows.append(
            {
                "dataset_name": dataset_name_raw,
                "dataset_name_safe": dataset_name_safe,
                "cell_id": cell_files.cell_id,
                "cycle_index": int(cycle_index),
                "mode": mode,
                "anode": common_cell_meta.get("anode"),
                "cathode": common_cell_meta.get("cathode"),
                "source": common_cell_meta.get("source"),
                "temperature_C": common_cell_meta.get("temperature_C"),
                "nominal_capacity_Ah": common_cell_meta.get("nominal_capacity_Ah"),
                "sample_id": sample_id,
                "sample_dir": sample_dir,
                "curve_path": curve_path,
                "meta_path": meta_path,
                "charge_png": charge_png,
                "discharge_png": discharge_png,
                "charge_png_std": charge_png_std,
                "discharge_png_std": discharge_png_std,
                "status": "ok",
                "is_complete": True,
                "completeness_reason": "ok",
                "fallback_steps": int(fallback_steps),
                "phase_inferred": bool(phase_inferred2),
                "phase_inferred_method": phase_method,
            }
        )

    if mode == "last_only":
        fallback_steps = 0
        selected = None
        selected_reason = None
        selected_phase_inferred = True

        for cyc in sorted(cycles, reverse=True):
            df_cycle = ts.loc[ts["cycle_index"].astype(int) == int(cyc)].copy()
            cycle_stats = _get_cycle_stats(cycle_df, int(cyc))
            ok, reason, df_charge, df_discharge, phase_inferred, phase_method = is_complete_cycle(
                df_cycle,
                cycle_stats=cycle_stats,
                min_points=min_points,
            )
            if ok:
                selected = int(cyc)
                selected_reason = reason
                selected_phase_inferred = phase_inferred
                break
            fallback_steps += 1

        if selected is None:
            index_rows.append(
                {
                    "dataset_name": dataset_name_raw,
                    "dataset_name_safe": dataset_name_safe,
                    "cell_id": cell_files.cell_id,
                    "mode": mode,
                    "status": "failed",
                    "fail_reason": "no_complete_cycle_found",
                }
            )
            return index_rows

        _export_one(cycle_index=selected, fallback_steps=fallback_steps)
        return index_rows

    if mode == "all_cycles":
        for cyc in cycles:
            _export_one(cycle_index=int(cyc), fallback_steps=0)
        return index_rows

    raise ValueError(f"Unknown mode: {mode}")


def run_dataset(
    root_dir: str,
    dataset_folder_name: str,
    out_dir: str,
    mode: str,
    min_points: int,
    overwrite: bool,
    max_cells: Optional[int],
    debug: bool,
) -> None:
    """Run one dataset folder under <root_dir>/Battery_Archive/<dataset_folder_name>."""

    dataset_name_raw = dataset_folder_name
    dataset_name_safe = safe_filename(dataset_folder_name)

    dataset_dir = os.path.join(root_dir, "Battery_Archive", dataset_folder_name)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")

    out_dataset_dir = os.path.join(out_dir, dataset_name_safe)
    ensure_dir(out_dataset_dir)

    # dataset_info.xlsx is used only for meta enrichment.
    dataset_info_path = os.path.join(root_dir, "Battery_Archive", "dataset_info.xlsx")
    _, meta_by_cell = read_dataset_info_xlsx(dataset_info_path, debug=debug)

    cells = discover_flat_cell_files(dataset_dir)
    if max_cells is not None:
        cells = cells[: int(max_cells)]

    index_rows: List[Dict[str, Any]] = []
    for cell in cells:
        try:
            rows = extract_samples_for_cell(
                dataset_name_raw=dataset_name_raw,
                dataset_name_safe=dataset_name_safe,
                cell_files=cell,
                meta_by_cell=meta_by_cell,
                out_dataset_dir=out_dataset_dir,
                mode=mode,
                min_points=min_points,
                overwrite=overwrite,
                debug=debug,
            )
            index_rows.extend(rows)
        except Exception as e:
            LOGGER.exception("Cell processing failed: dataset=%s cell=%s", dataset_name_raw, cell.cell_id)
            index_rows.append(
                {
                    "dataset_name": dataset_name_raw,
                    "dataset_name_safe": dataset_name_safe,
                    "cell_id": cell.cell_id,
                    "mode": mode,
                    "status": "failed",
                    "fail_reason": f"unhandled:{type(e).__name__}:{e}",
                }
            )

    index_df = pd.DataFrame(index_rows)

    index_xlsx = os.path.join(out_dataset_dir, "index_samples.xlsx")
    index_csv = os.path.join(out_dataset_dir, "index_samples.csv")

    with pd.ExcelWriter(index_xlsx, engine="openpyxl") as writer:
        index_df.to_excel(writer, index=False, sheet_name="samples")

    index_df.to_csv(index_csv, index=False)

    # Summary
    summary: Dict[str, Any] = {
        "dataset_name": dataset_name_raw,
        "dataset_name_safe": dataset_name_safe,
        "dataset_dir": dataset_dir,
        "out_dataset_dir": out_dataset_dir,
        "mode": mode,
        "min_points": int(min_points),
        "total_cells": int(len(cells)),
        "total_rows": int(len(index_df)),
        "ok_samples": int((index_df.get("status") == "ok").sum()) if "status" in index_df.columns else 0,
        "failed_cells": int((index_df.get("status") == "failed").sum()) if "status" in index_df.columns else 0,
    }

    if "fail_reason" in index_df.columns:
        top_fail = (
            index_df.loc[index_df["status"] == "failed", "fail_reason"]
            .value_counts()
            .head(10)
            .to_dict()
        )
        summary["top_fail_reasons"] = top_fail

    if "fallback_steps" in index_df.columns:
        fb = index_df.loc[index_df["status"] == "ok", "fallback_steps"].dropna()
        summary["fallback_steps_stats"] = {
            "count": int(len(fb)),
            "mean": float(fb.mean()) if len(fb) else None,
            "max": int(fb.max()) if len(fb) else None,
        }

    write_json(os.path.join(out_dataset_dir, "summary.json"), summary)


def discover_dataset_folders(root_dir: str) -> List[str]:
    base = os.path.join(root_dir, "Battery_Archive")
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Battery_Archive folder not found: {base}")

    out = []
    for name in os.listdir(base):
        full = os.path.join(base, name)
        if not os.path.isdir(full):
            continue
        if name.startswith("__"):
            continue
        out.append(name)

    # Keep stable order
    out = sorted(out)

    # Filter out folders that do not look like datasets
    # (e.g. __MACOSX is already filtered; keep all others)
    return out


def run_single_dataset_cli(dataset_folder_name: str) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Extract single-observation features for one Battery Archive dataset")
    parser.add_argument("--root_dir", default=r"F:\Battery\dataset\Tao\Battery_Archive")
    parser.add_argument("--out_dir", default=r"F:\Battery\dataset\Tao\Battery_Archive\outputs\features")
    parser.add_argument("--mode", default="last_only", choices=["last_only", "all_cycles"])
    parser.add_argument("--min_points", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_cells", type=int, default=None)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    setup_logging(args.debug)

    run_dataset(
        root_dir=args.root_dir,
        dataset_folder_name=dataset_folder_name,
        out_dir=args.out_dir,
        mode=args.mode,
        min_points=args.min_points,
        overwrite=bool(args.overwrite),
        max_cells=args.max_cells,
        debug=args.debug,
    )
