#!/usr/bin/env python
# -*- coding: utf-8 -*-

# === 脚本用途 ===
# - 全功能 AMOTF + DINO SOH：build_amotf / train / run_all
# - 默认输入 amotf_npz；train 亦支持 amotf、png
# - 若仅需「全量数据」或「dataset 组合 sweep」，可改用同目录自包含脚本:
#     soh_dino_amotf_npz_soc_horizontal.py
#     soh_dino_amotf_npz_dataset_cycles_all_classes_soc_horizontal.py
#
# === 通用说明 ===
# - train 前需已有对应 npz；全量 npz 由 build_amotf 生成。
# - 续行使用反斜杠 \ 时，该行末尾不得再跟其它字符（否则 shell 会把下一行粘坏）。
# - 离线推理训练可加: --hf_local_only（需已缓存 HuggingFace 权重）。
#
# === 命令行（路径可按机器修改；续行 \ 后勿再跟字符）===
#
# --- 帮助 ---
# python /mnt/sdb/THX/Battery_THX_HP_P9000/Battery/dataset/Tao/Battery_Archive/scripts/battery-soh-dino/train/soh_dino_amotf_npz_horiz_train_core.py --help
#
# --- build_amotf ---
# python /mnt/sdb/THX/Battery_THX_HP_P9000/Battery/dataset/Tao/Battery_Archive/scripts/battery-soh-dino/train/soh_dino_amotf_npz_horiz_train_core.py build_amotf \
#   --labels_csv /mnt/sdb/THX/Battery_THX_HP_P9000/no_title_outputs/features/soh_classification_results.csv \
#   --max_points 2000 \
#   --m 5 --tau 1 --spans 1,2,4,8 \
#   --out_size 0 --overwrite --num_workers 4
#
# --- train（全量 amotf_npz；子命令使用 --epochs）---
# python /mnt/sdb/THX/Battery_THX_HP_P9000/Battery/dataset/Tao/Battery_Archive/scripts/battery-soh-dino/train/soh_dino_amotf_npz_horiz_train_core.py train \
#   --labels_csv /mnt/sdb/THX/Battery_THX_HP_P9000/no_title_outputs/features/soh_classification_results.csv \
#   --runs_root /mnt/sdb/THX/Battery_THX_HP_P9000/no_title_outputs/soh_amotf_dino_runs \
#   --run_name exp_train_npz \
#   --input_mode amotf_npz \
#   --max_points 2000 \
#   --kfold 5 \
#   --epochs 50 \
#   --batch_size 24 \
#   --lr 5e-4 \
#   --weight_decay 1e-4 \
#   --lr_scheduler cosine_warmup \
#   --lr_warmup_ratio 0.1 \
#   --lr_min 1e-6 \
#   --backbone_lr_mult 0.1 \
#   --finetune_backbone \
#   --use_class_weights \
#   --npz_norm log1p_global \
#   --npz_global_max_log 10.0 \
#   --num_workers 4
#
# --- run_all ---
# python /mnt/sdb/THX/Battery_THX_HP_P9000/Battery/dataset/Tao/Battery_Archive/scripts/battery-soh-dino/train/soh_dino_amotf_npz_horiz_train_core.py run_all \
#   --labels_csv /mnt/sdb/THX/Battery_THX_HP_P9000/no_title_outputs/features/soh_classification_results.csv \
#   --runs_root /mnt/sdb/THX/Battery_THX_HP_P9000/no_title_outputs/soh_amotf_dino_runs \
#   --run_name exp_run_all \
#   --max_points 2000 \
#   --epochs 50 \
#   --finetune_backbone \
#   --use_class_weights \
#   --lr_scheduler cosine_warmup \
#   --lr_warmup_ratio 0.1 \
#   --lr_min 1e-6 \
#   --backbone_lr_mult 0.1 \
#   --npz_norm log1p_global \
#   --npz_global_max_log 10.0 \
#   --num_workers_build 4 \
#   --num_workers 4



 # 备份版本, 横轴为SOC, 纵轴为电压和电流, 将电压和电流(充放电电压+充放电电流)用amotf转为图像后用dino分类
 # 目前最好结果: /media/haoxiang/THX_HP_P900/Battery/dataset/Tao/Battery_Archive/outputs/soh_amotf_dino_runs/run_20260124_014958

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


LOGGER = logging.getLogger("soh_amotf_dino")


ALLOWED_DATASETS = [
    "CALCE",
    "HNEI",
    "Michigan_Expansion",
    "Michigan_Formation",
    "Oxford",
    "SNL_LFP",
    "SNL_NCA",
    "SNL_NMC",
    "UL_Purdue",
]

LABEL_MAP = {"A": 0, "B": 1, "C": 2}
INV_LABEL = {v: k for k, v in LABEL_MAP.items()}


# Cross-OS path handling (Windows F:\ -> Ubuntu /media/haoxiang/THX_HP_P900/ and vice versa)
PLATFORM_IS_WIN = os.name == "nt"
KNOWN_WIN_ROOT = r"F:\\"
KNOWN_LINUX_ROOT = "/media/haoxiang/THX_HP_P900"

KNOWN_LINUX_ROOTS = [
    "/mnt/sdb/THX/Battery_THX_HP_P9000",
    KNOWN_LINUX_ROOT,
]

def _split_all_parts(p: str) -> Tuple[str, ...]:
    return tuple([q for q in re.split(r"[\\/]+", str(p)) if q])

def remap_known_root(p: str) -> str:
    s = str(p).strip()
    if not s:
        return s
    if PLATFORM_IS_WIN:
        # Map Linux mount root to Windows drive
        for root in KNOWN_LINUX_ROOTS:
            if s.startswith(root):
                rest = s[len(root):].lstrip("/\\")
                out = KNOWN_WIN_ROOT + rest
                return out.replace("/", "\\")
        return s.replace("/", "\\")
    else:
        # POSIX: map F:\... -> /mnt/.../..., prefer an existing mount
        m = re.match(r"^([A-Za-z]):[\\/](.*)$", s)
        if m:
            drive = m.group(1).upper()
            if drive == "F":
                rest = m.group(2)
                tail = rest.replace("\\", "/")
                for root in KNOWN_LINUX_ROOTS:
                    out = root.rstrip("/\\") + "/" + tail
                    if os.path.exists(out):
                        return out
                return KNOWN_LINUX_ROOTS[0].rstrip("/\\") + "/" + tail
        out = s.replace("\\", "/")

        for root in KNOWN_LINUX_ROOTS:
            r = str(root).rstrip("/\\")
            old_feat = r + "/Battery/dataset/Tao/Battery_Archive/outputs/features"
            new_feat = r + "/no_title_outputs/features"
            if out.startswith(old_feat):
                tail = out[len(old_feat):].lstrip("/")
                cand = new_feat + ("/" + tail if tail else "")
                try:
                    if os.path.exists(cand):
                        return cand
                except Exception:
                    pass
        return out

def _platform_root() -> str:
    if PLATFORM_IS_WIN:
        return KNOWN_WIN_ROOT
    for root in KNOWN_LINUX_ROOTS:
        if os.path.exists(root):
            return root
    return KNOWN_LINUX_ROOTS[0]

def _default_labels_csv() -> str:
    return os.path.join(
        _platform_root(),
        "Battery",
        "dataset",
        "Tao",
        "Battery_Archive",
        "outputs",
        "soh_classification_results.csv",
    )

def _default_runs_root() -> str:
    return os.path.join(
        _platform_root(),
        "Battery",
        "dataset",
        "Tao",
        "Battery_Archive",
        "outputs",
        "soh_amotf_dino_runs",
    )


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _setup_logging(log_path: Path, level: str = "INFO") -> None:
    _ensure_dir(log_path.parent)
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    LOGGER.setLevel(lvl)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(lvl)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(lvl)
    sh.setFormatter(fmt)

    LOGGER.handlers.clear()
    LOGGER.addHandler(fh)
    LOGGER.addHandler(sh)


# ----------------------------
# Public APIs (to be filled)
# ----------------------------


def _tqdm(it: Iterable, *, total: Optional[int] = None, desc: str = ""):
    try:
        from tqdm import tqdm

        return tqdm(it, total=total, desc=desc)
    except Exception:
        return it


def _try_import_pandas():
    try:
        import pandas as pd

        return pd
    except Exception:
        return None


def _try_import_sklearn():
    try:
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
        from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

        return StratifiedKFold, StratifiedShuffleSplit, accuracy_score, confusion_matrix, f1_score
    except Exception:
        return None


def _ensure_sklearn():
    try:
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
        from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

        return StratifiedKFold, StratifiedShuffleSplit, accuracy_score, confusion_matrix, f1_score
    except Exception as e:
        raise RuntimeError(
            "scikit-learn is required for joint-stratified k-fold and metrics. "
            "Install it with either: pip install scikit-learn  OR  conda install scikit-learn. "
            f"import_error={e}"
        )


def _now_run_name() -> str:
    return time.strftime("run_%Y%m%d_%H%M%S", time.localtime())


def _sha1_short(s: str, n: int = 12) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:n]


def infer_dataset_name_from_original_path(original_path: str) -> Optional[str]:
    s = remap_known_root(original_path)
    parts = _split_all_parts(s)
    allowed_norm = {re.sub(r"[\s\-]+", "_", str(x)).lower(): str(x) for x in ALLOWED_DATASETS}
    for p in parts:
        pn = re.sub(r"[\s\-]+", "_", str(p)).lower()
        if pn in allowed_norm:
            return allowed_norm[pn]
    return None


def _exists(p: str) -> bool:
    try:
        q = remap_known_root(p)
        return isinstance(q, str) and len(q) > 0 and os.path.exists(q)
    except Exception:
        return False


def _is_valid_npz(p: str) -> bool:
    try:
        q = remap_known_root(p)
        if not isinstance(q, str) or not q:
            return False
        if not os.path.exists(q):
            return False
        if os.path.getsize(q) <= 0:
            return False
        if not zipfile.is_zipfile(q):
            return False
        with np.load(q, allow_pickle=False) as z:
            if "amotf" not in z.files:
                return False
        try:
            with zipfile.ZipFile(q, "r") as zf:
                names = zf.namelist()
                if not any(str(n).endswith("amotf.npy") for n in names):
                    return False
        except Exception:
            return False
        return True
    except Exception:
        return False


def _p_or_missing(p: str) -> str:
    return p if _exists(p) else "MISSING"


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _try_import_pil_image():
    try:
        from PIL import Image

        return Image
    except Exception:
        return None


def stable_argsort(a: np.ndarray) -> np.ndarray:
    return np.argsort(a, kind="stable")

# 序数编码
def permutation_to_index(perm: np.ndarray) -> int:
    m = len(perm)
    code = 0
    for i in range(m):
        c = 0
        pi = perm[i]
        for j in range(i + 1, m):
            if perm[j] < pi:
                c += 1
        code = code * (m - i) + c
    return int(code)

# 获取windows来序数编码
def get_permutation_index(window: np.ndarray) -> int:
    idx = stable_argsort(window)
    return permutation_to_index(idx)

# 计算emd权重
def calculate_emd_weight(window_a_sorted: np.ndarray, window_b_sorted: np.ndarray) -> float:
    m = int(window_a_sorted.size)
    if m <= 0:
        return 1.0
    emd = np.abs(window_a_sorted - window_b_sorted).mean()
    return 1.0 + float(emd)


def build_amotf_single_modality(series: np.ndarray, m: int, tau: int, span_set: List[int]) -> np.ndarray:
    # 预处理,去除异常值
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    # 计算模式数
    num_patterns = math.factorial(int(m))
    # 窗口长度
    if n < (m - 1) * tau + 1:
        return np.zeros((num_patterns, num_patterns, len(span_set)), dtype=np.float32)
    # 构造延迟嵌入窗口（ordinal windows）
    num_windows = n - (m - 1) * tau
    windows = np.stack([x[i : i + m * tau : tau] for i in range(num_windows)], axis=0)
    # 把窗口映射成“排列模式索引”
    patterns = np.array([get_permutation_index(w) for w in windows], dtype=np.int32)
    # 排序窗口（为权重计算做准备）
    windows_sorted = np.sort(windows, axis=1, kind="stable")
    # 初始化输出张量
    out = np.zeros((num_patterns, num_patterns, len(span_set)), dtype=np.float32)

    # amotf 算法
    for s_idx, s in enumerate(span_set):
        if s <= 0:
            continue
        max_t = num_windows - 1 - s
        if max_t < 0:
            continue
        mat = out[:, :, s_idx]
        for t in range(0, max_t + 1):
            # 计算转移模式
            src = int(patterns[t])
            tgt = int(patterns[t + s])
            # 计算转移权重
            w_src = windows_sorted[t]
            w_tgt = windows_sorted[t + s]
            weight = calculate_emd_weight(w_src, w_tgt)
            mat[src, tgt] += weight
    return out

# _amotf_worker 封装
def _amotf_worker(payload: Tuple[str, int, int, int, List[int], int, bool]) -> Tuple[bool, str]:
    sample_dir, out_size, m, tau, spans, max_points, overwrite = payload
    return build_and_save_amotf_for_sample(
        sample_dir,
        out_size=int(out_size),
        m=int(m),
        tau=int(tau),
        spans=list(spans),
        max_points=int(max_points),
        overwrite=bool(overwrite),
    )

# 对v和i两个方式分别建立 amotf 算法
def build_amotf_multimodal(voltage: np.ndarray, current: np.ndarray, m: int, tau: int, span_set: List[int]) -> np.ndarray:
    av = build_amotf_single_modality(voltage, m, tau, span_set)
    ai = build_amotf_single_modality(current, m, tau, span_set)
    return np.concatenate([av, ai], axis=2)

# 转为为uint 8图像, 最大值最小值归一化方式,可能为问题根因 !
def to_uint8_img(x: np.ndarray, per_channel: bool = True) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x[..., None]
    eps = 1e-8
    y = np.zeros_like(x)
    if per_channel:
        for c in range(int(x.shape[2])):
            v = x[:, :, c]
            vmin = float(np.min(v))
            vmax = float(np.max(v))
            if vmax - vmin < eps:
                y[:, :, c] = 0.0
            else:
                y[:, :, c] = (v - vmin) / (vmax - vmin)
    else:
        vmin = float(np.min(x))
        vmax = float(np.max(x))
        if vmax - vmin < eps:
            y = np.zeros_like(x)
        else:
            y = (x - vmin) / (vmax - vmin)
    y = np.clip(y * 255.0, 0, 255).astype(np.uint8)
    return y

# 从多通道中重建rgb图像 !
def make_rgb_from_multichan(tensor_hwk: np.ndarray) -> np.ndarray:
    H, W, K = tensor_hwk.shape
    half = int(K // 2)
    v = tensor_hwk[:, :, :half]
    i = tensor_hwk[:, :, half:]
    R = v.mean(axis=2)
    G = i.mean(axis=2)
    B = 0.5 * (R + G)
    rgb = np.stack([R, G, B], axis=2)
    return to_uint8_img(rgb, per_channel=True)

# 保存uint8图像
def save_png_uint8(arr_uint8_hw3: np.ndarray, path: Path, out_size: int = 224) -> None:
    Image = _try_import_pil_image()
    if Image is None:
        raise RuntimeError("Pillow is required to save PNG")
    _ensure_dir(path.parent)
    im = Image.fromarray(arr_uint8_hw3, mode="RGB")
    if int(out_size) > 0 and (im.size[0] != int(out_size) or im.size[1] != int(out_size)):
        im = im.resize((int(out_size), int(out_size)), resample=Image.BICUBIC)
    im.save(str(path))


# 从已有列名中，根据一组模式（精确匹配优先，其次正则匹配），挑选最合适的一列(读文件)
def _pick_col(cols: Sequence[str], patterns: Sequence[str]) -> Optional[str]:
    cols_s = [str(c) for c in cols]
    lower_map = {c.lower(): c for c in cols_s}
    for p in patterns:
        if p.lower() in lower_map:
            return lower_map[p.lower()]
    for p in patterns:
        rg = re.compile(p, flags=re.IGNORECASE)
        for c in cols_s:
            if rg.search(c):
                return c
    return None


# 横轴为SOC, 纵轴为电压或者电流
def read_curve_csv(curve_csv: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(curve_csv):
        return None

    pd = _try_import_pandas()
    if pd is not None:
        try:
            df = pd.read_csv(curve_csv)
        except Exception:
            return None
        cols = list(df.columns)
        time_col = _pick_col(cols, [r"^t_s$", r"^t$", r"time", r"test_time", r"seconds", r"\(s\)"])
        soc_col = _pick_col(cols, [r"^soc_pct$"])
        volt_col = _pick_col(cols, [r"voltage", r"^v$", r"voltage_v"])
        curr_col = _pick_col(cols, [r"current", r"^i$", r"current_a", r"current_abs_a"])
        phase_col = _pick_col(cols, [r"^phase$"])
        if volt_col is None or curr_col is None:
            return None
        return {
            "backend": "pandas",
            "df": df,
            "time_col": time_col,
            "soc_col": soc_col,
            "volt_col": volt_col,
            "curr_col": curr_col,
            "phase_col": phase_col,
        }

    with open(curve_csv, "r", encoding="utf-8", errors="replace", newline="") as f:
        rdr = csv.DictReader(f)
        cols = rdr.fieldnames or []
        time_col = _pick_col(cols, [r"^t_s$", r"^t$", r"time", r"test_time", r"seconds", r"\(s\)"])
        soc_col = _pick_col(cols, [r"^soc_pct$"])
        volt_col = _pick_col(cols, [r"voltage", r"^v$", r"voltage_v"])
        curr_col = _pick_col(cols, [r"current", r"^i$", r"current_a", r"current_abs_a"])
        phase_col = _pick_col(cols, [r"^phase$"])
        if volt_col is None or curr_col is None:
            return None
        rows = [dict(r) for r in rdr]
    return {
        "backend": "csv",
        "rows": rows,
        "time_col": time_col,
        "soc_col": soc_col,
        "volt_col": volt_col,
        "curr_col": curr_col,
        "phase_col": phase_col,
    }


def split_charge_discharge(curve: Dict[str, Any], min_points: int = 30) -> Tuple[
     Optional[Tuple[np.ndarray, np.ndarray]],
     Optional[Tuple[np.ndarray, np.ndarray]],
     str,
 ]:
     backend = curve.get("backend")
     time_col = curve.get("time_col")
     volt_col = curve.get("volt_col")
     curr_col = curve.get("curr_col")
     phase_col = curve.get("phase_col")

     if backend == "pandas":
         pd = _try_import_pandas()
         df = curve["df"].copy()
         if time_col is not None and time_col in df.columns:
             df[time_col] = pd.to_numeric(df[time_col], errors="coerce").astype(float)
         else:
             time_col = "__t"
             df[time_col] = np.arange(len(df), dtype=float)
         df[volt_col] = pd.to_numeric(df[volt_col], errors="coerce").astype(float)
         df[curr_col] = pd.to_numeric(df[curr_col], errors="coerce").astype(float)
         df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[time_col, volt_col, curr_col])
         if df.empty or len(df) < 2:
             return None, None, "no_valid_rows"
         df = df.sort_values(time_col, kind="stable")

         # 基于标签区分充电和放电
         if phase_col is not None and phase_col in df.columns:
             ph = df[phase_col].astype(str)
             df_c = df.loc[ph == "charge"]
             df_d = df.loc[ph == "discharge"]
             if len(df_c) >= min_points and len(df_d) >= min_points:
                 return (
                     df_c[volt_col].to_numpy(dtype=float),
                     df_c[curr_col].to_numpy(dtype=float),
                 ), (
                     df_d[volt_col].to_numpy(dtype=float),
                     df_d[curr_col].to_numpy(dtype=float),
                 ), "phase"

         # 否则用电压高点来判断充放电
         vv = df[volt_col].to_numpy(dtype=float)
         ii = df[curr_col].to_numpy(dtype=float)
         k = int(np.nanargmax(vv))
         c = (vv[: k + 1], ii[: k + 1])
         d = (vv[k + 1 :], ii[k + 1 :])
         if len(c[0]) < min_points or len(d[0]) < min_points:
             return None, None, "phase_split_failed"
         return c, d, "fallback_maxV"

     def _to_float(x: Any) -> float:
         try:
             return float(x)
         except Exception:
             return float("nan")

     rows: List[Dict[str, Any]] = curve.get("rows") or []
     items: List[Tuple[float, float, float, str]] = []
     for idx, r in enumerate(rows):
         t = _to_float(r.get(time_col)) if time_col else float(idx)
         v = _to_float(r.get(volt_col))
         i = _to_float(r.get(curr_col))
         ph = str(r.get(phase_col)) if phase_col else ""
         if np.isfinite(t) and np.isfinite(v) and np.isfinite(i):
             items.append((t, v, i, ph))
     if len(items) < 2:
         return None, None, "too_few_points"
     items.sort(key=lambda x: x[0])
     if phase_col:
         c2 = [(v, i) for (_, v, i, ph) in items if ph == "charge"]
         d2 = [(v, i) for (_, v, i, ph) in items if ph == "discharge"]
         if len(c2) >= min_points and len(d2) >= min_points:
             return (
                 np.array([x[0] for x in c2], dtype=float),
                 np.array([x[1] for x in c2], dtype=float),
             ), (
                 np.array([x[0] for x in d2], dtype=float),
                 np.array([x[1] for x in d2], dtype=float),
             ), "phase"
     vv = np.array([x[1] for x in items], dtype=float)
     ii = np.array([x[2] for x in items], dtype=float)
     k = int(np.nanargmax(vv))
     c = (vv[: k + 1], ii[: k + 1])
     d = (vv[k + 1 :], ii[k + 1 :])
     if len(c[0]) < min_points or len(d[0]) < min_points:
         return None, None, "phase_split_failed"
     return c, d, "fallback_maxV"


def split_charge_discharge_with_soc(curve: Dict[str, Any], min_points: int = 30) -> Tuple[
     Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
     Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
     str,
 ]:
     backend = curve.get("backend")
     time_col = curve.get("time_col")
     soc_col = curve.get("soc_col")
     volt_col = curve.get("volt_col")
     curr_col = curve.get("curr_col")
     phase_col = curve.get("phase_col")

     if soc_col is None:
         return None, None, "missing_soc_col"

     if backend == "pandas":
         pd = _try_import_pandas()
         df = curve["df"].copy()
         if time_col is not None and time_col in df.columns:
             df[time_col] = pd.to_numeric(df[time_col], errors="coerce").astype(float)
         else:
             time_col = "__t"
             df[time_col] = np.arange(len(df), dtype=float)
         df[volt_col] = pd.to_numeric(df[volt_col], errors="coerce").astype(float)
         df[curr_col] = pd.to_numeric(df[curr_col], errors="coerce").astype(float)
         df[soc_col] = pd.to_numeric(df[soc_col], errors="coerce").astype(float)
         df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[time_col, volt_col, curr_col, soc_col])
         if df.empty or len(df) < 2:
             return None, None, "no_valid_rows"
         df = df.sort_values(time_col, kind="stable")

         if phase_col is not None and phase_col in df.columns:
             ph = df[phase_col].astype(str)
             df_c = df.loc[ph == "charge"]
             df_d = df.loc[ph == "discharge"]
             if len(df_c) >= min_points and len(df_d) >= min_points:
                 return (
                     df_c[soc_col].to_numpy(dtype=float),
                     df_c[volt_col].to_numpy(dtype=float),
                     df_c[curr_col].to_numpy(dtype=float),
                 ), (
                     df_d[soc_col].to_numpy(dtype=float),
                     df_d[volt_col].to_numpy(dtype=float),
                     df_d[curr_col].to_numpy(dtype=float),
                 ), "phase"

         ss = df[soc_col].to_numpy(dtype=float)
         vv = df[volt_col].to_numpy(dtype=float)
         ii = df[curr_col].to_numpy(dtype=float)
         k = int(np.nanargmax(vv))
         c = (ss[: k + 1], vv[: k + 1], ii[: k + 1])
         d = (ss[k + 1 :], vv[k + 1 :], ii[k + 1 :])
         if len(c[0]) < min_points or len(d[0]) < min_points:
             return None, None, "phase_split_failed"
         return c, d, "fallback_maxV"

     def _to_float(x: Any) -> float:
         try:
             return float(x)
         except Exception:
             return float("nan")

     rows: List[Dict[str, Any]] = curve.get("rows") or []
     items: List[Tuple[float, float, float, float, str]] = []
     for idx, r in enumerate(rows):
         t = _to_float(r.get(time_col)) if time_col else float(idx)
         s = _to_float(r.get(soc_col))
         v = _to_float(r.get(volt_col))
         i = _to_float(r.get(curr_col))
         ph = str(r.get(phase_col)) if phase_col else ""
         if np.isfinite(t) and np.isfinite(s) and np.isfinite(v) and np.isfinite(i):
             items.append((t, s, v, i, ph))
     if len(items) < 2:
         return None, None, "too_few_points"
     items.sort(key=lambda x: x[0])

     if phase_col:
         c2 = [(s, v, i) for (_, s, v, i, ph) in items if str(ph) == "charge"]
         d2 = [(s, v, i) for (_, s, v, i, ph) in items if str(ph) == "discharge"]
         if len(c2) >= min_points and len(d2) >= min_points:
             return (
                 np.array([x[0] for x in c2], dtype=float),
                 np.array([x[1] for x in c2], dtype=float),
                 np.array([x[2] for x in c2], dtype=float),
             ), (
                 np.array([x[0] for x in d2], dtype=float),
                 np.array([x[1] for x in d2], dtype=float),
                 np.array([x[2] for x in d2], dtype=float),
             ), "phase"

     ss = np.array([x[1] for x in items], dtype=float)
     vv = np.array([x[2] for x in items], dtype=float)
     ii = np.array([x[3] for x in items], dtype=float)
     k = int(np.nanargmax(vv))
     c = (ss[: k + 1], vv[: k + 1], ii[: k + 1])
     d = (ss[k + 1 :], vv[k + 1 :], ii[k + 1 :])
     if len(c[0]) < min_points or len(d[0]) < min_points:
         return None, None, "phase_split_failed"
     return c, d, "fallback_maxV"


def resample_series_by_x(x: np.ndarray, y: np.ndarray, max_points: int = 2000) -> np.ndarray:
     x = np.asarray(x, dtype=float)
     y = np.asarray(y, dtype=float)
     m = int(max_points)
     mask = np.isfinite(x) & np.isfinite(y)
     x = x[mask]
     y = y[mask]
     if m <= 0:
         return y
     if x.size == 0:
         return np.zeros((m,), dtype=float)
     if x.size == 1:
         return np.full((m,), float(y[0]), dtype=float)

     # 保持横轴方向：如果 SOC 递减（常见于 discharge），则输出也保持递减
     reverse_out = False
     if float(x[0]) > float(x[-1]):
         x = x[::-1]
         y = y[::-1]
         reverse_out = True

     order = np.argsort(x, kind="stable")
     x = x[order]
     y = y[order]
     uniq_x, _ = np.unique(x, return_index=True)
     if uniq_x.size != x.size:
         means = []
         for ux in uniq_x:
             sel = (x == ux)
             means.append(float(np.mean(y[sel])))
         x = uniq_x
         y = np.array(means, dtype=float)
     if x.size == 1:
         out = np.full((m,), float(y[0]), dtype=float)
         return out[::-1] if reverse_out else out

     xi = np.linspace(float(x[0]), float(x[-1]), num=m)
     yi = np.interp(xi, x, y).astype(float)
     return yi[::-1] if reverse_out else yi


def resample_series(x: np.ndarray, max_points: int = 2000) -> np.ndarray:
     x = np.asarray(x, dtype=float)
     x = x[np.isfinite(x)]
     tgt = int(max_points)
     if tgt <= 0:
         return x
     if x.size == 0:
         return x
     if x.size == 1:
         return np.full((tgt,), float(x[0]), dtype=float)
     if x.size == tgt:
         return x

     # 线性插值
     idx = np.linspace(0, x.size - 1, num=tgt)
     i0 = np.floor(idx).astype(int)
     i1 = np.minimum(i0 + 1, x.size - 1)
     w = idx - i0
     return (1.0 - w) * x[i0] + w * x[i1]

# 最小值归一化到0，最大值归一化到1，其他值线性插值
def _norm_minmax(x: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(x)
    if not bool(np.any(mask)):
        return np.zeros_like(x, dtype=float)
    vmin = float(np.min(x[mask]))
    vmax = float(np.max(x[mask]))
    denom = float(vmax - vmin)
    if denom < float(eps):
        return np.zeros_like(x, dtype=float)
    y = (x - vmin) / denom
    y = np.where(mask, y, 0.0)
    return np.clip(y, 0.0, 1.0).astype(float)

# 最大绝对值归一化到1，其他值线性插值
def _norm_maxabs(x: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(x)
    if not bool(np.any(mask)):
        return np.zeros_like(x, dtype=float)
    m = float(np.max(np.abs(x[mask])))
    if m < float(eps):
        return np.zeros_like(x, dtype=float)
    y = x / m
    y = np.where(mask, y, 0.0)
    return y.astype(float)

# 读取标签文件，返回包含sample_id, original_path, assigned_class的字典列表
def load_label_rows(labels_csv: str) -> List[Dict[str, Any]]:
    labels_csv = remap_known_root(labels_csv)
    if not os.path.exists(labels_csv):
        raise FileNotFoundError(labels_csv)
    pd = _try_import_pandas()
    if pd is not None:
        df = pd.read_csv(labels_csv)
        rows = df.to_dict(orient="records")
    else:
        with open(labels_csv, "r", encoding="utf-8", errors="replace", newline="") as f:
            rows = [dict(r) for r in csv.DictReader(f)]
    out: List[Dict[str, Any]] = []
    for r in rows:
        op = str(r.get("original_path", "")).strip()
        lab = str(r.get("assigned_class", "")).strip()
        if not op or not lab:
            continue
        out.append({"sample_id": str(r.get("sample_id", "")).strip(), "original_path": op, "assigned_class": lab})
    return out


def _base_sample_id(sample_id: str) -> str:
    s = str(sample_id or "").strip()
    if not s:
        return s
    s = re.sub(r"(_(charge|discharge))?_p\d+$", "", s)
    s = re.sub(r"_p\d+$", "", s)
    return s


def _default_exclude_samples_txt() -> str:
    for root in KNOWN_LINUX_ROOTS:
        cand = os.path.join(str(root).rstrip("/\\"), "no_title_outputs", "exclude_samples.txt")
        try:
            if os.path.exists(cand):
                return cand
        except Exception:
            continue
    return os.path.join(str(KNOWN_LINUX_ROOTS[0]).rstrip("/\\"), "no_title_outputs", "exclude_samples.txt")


def _load_exclude_sets(exclude_samples_txt: str, *, labels_csv: str) -> Tuple[set, set]:
    p = remap_known_root(str(exclude_samples_txt or "").strip())
    if (not p) or (not os.path.exists(p)):
        return set(), set()

    raw_sids: set = set()
    raw_paths: set = set()
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = str(line).strip()
            if not s:
                continue
            if s.startswith("#"):
                continue
            s = remap_known_root(s)

            if ("/" in s) or ("\\" in s):
                if s.lower().endswith(".png"):
                    raw_sids.add(os.path.splitext(os.path.basename(s))[0])
                else:
                    raw_paths.add(s)
                    raw_sids.add(os.path.basename(s.rstrip("/\\")))
            else:
                raw_sids.add(s)

    rows = load_label_rows(labels_csv)
    sid2op: Dict[str, str] = {}
    for r in rows:
        sid = _base_sample_id(str(r.get("sample_id", "")))
        op = remap_known_root(str(r.get("original_path", "")).strip())
        if sid and op:
            sid2op[sid] = op

    exclude_sid_base: set = set([_base_sample_id(x) for x in raw_sids if str(x).strip()])
    exclude_paths: set = set([remap_known_root(x) for x in raw_paths if str(x).strip()])
    for sid in list(exclude_sid_base):
        op = sid2op.get(str(sid))
        if op:
            exclude_paths.add(op)
    return exclude_sid_base, exclude_paths

# 生成amotf输出文件名后缀，用于区分不同采样点数的amotf结果文件
def _amotf_name_suffix(max_points: int) -> str:
    mp = int(max_points)
    if mp == 2000:
        return ""
    return f"_mp{mp}"

# 生成amotf输出文件路径
def _amotf_output_paths(sample_dir: str, *, max_points: int) -> Dict[str, Path]:
    out_dir = Path(remap_known_root(sample_dir)) / "amotf"
    suf = _amotf_name_suffix(int(max_points))
    return {
        "out_dir": out_dir,
        "c_png": out_dir / f"amotf_charge{suf}.png",
        "d_png": out_dir / f"amotf_discharge{suf}.png",
        "c_npz": out_dir / f"amotf_charge{suf}.npz",
        "d_npz": out_dir / f"amotf_discharge{suf}.npz",
    }

# 构建amotf并保存结果文件
def build_and_save_amotf_for_sample(
    sample_dir: str,
    *,
    out_size: int,
    m: int,
    tau: int,
    spans: List[int],
    max_points: int,
    overwrite: bool,
) -> Tuple[bool, str]:
    curve_csv = os.path.join(sample_dir, "curve.csv")
    if not os.path.exists(curve_csv):
        return False, "missing_curve_csv"
    paths = _amotf_output_paths(sample_dir, max_points=int(max_points))
    out_dir = paths["out_dir"]
    c_png = paths["c_png"]
    d_png = paths["d_png"]
    c_npz = paths["c_npz"]
    d_npz = paths["d_npz"]
    if (
        (not overwrite)
        and c_png.exists()
        and d_png.exists()
        and _is_valid_npz(str(c_npz))
        and _is_valid_npz(str(d_npz))
    ):
        return True, "exists"

    curve = read_curve_csv(curve_csv)
    if curve is None:
        return False, "curve_parse_failed"
    c_trip, d_trip, why = split_charge_discharge_with_soc(curve)
    if c_trip is None or d_trip is None:
        return False, f"split_failed:{why}"

    # 分离充电和放电曲线（SOC, V, I）
    s_c, v_c, i_c = c_trip
    s_d, v_d, i_d = d_trip
    
    # 采样（横轴 SOC）
    v_c = resample_series_by_x(s_c, v_c, max_points=max_points)
    i_c = resample_series_by_x(s_c, i_c, max_points=max_points)
    v_d = resample_series_by_x(s_d, v_d, max_points=max_points)
    i_d = resample_series_by_x(s_d, i_d, max_points=max_points)
    
    # 归一化
    v_c = _norm_minmax(v_c)
    v_d = _norm_minmax(v_d)
    i_c = _norm_maxabs(i_c)
    i_d = _norm_maxabs(i_d)

    # 根据充电和放电分别构建amotf
    tens_c = build_amotf_multimodal(v_c, i_c, m=m, tau=tau, span_set=spans)
    tens_d = build_amotf_multimodal(v_d, i_d, m=m, tau=tau, span_set=spans)

    _ensure_dir(out_dir)
    np.savez_compressed(str(c_npz), amotf=tens_c.astype(np.float32))
    np.savez_compressed(str(d_npz), amotf=tens_d.astype(np.float32))

    # 生成RGB图像（三通道：电压，电流，取平均（！可以考虑换））
    rgb_c = make_rgb_from_multichan(tens_c)
    rgb_d = make_rgb_from_multichan(tens_d)
    save_png_uint8(rgb_c, c_png, out_size=out_size)
    save_png_uint8(rgb_d, d_png, out_size=out_size)

    return True, "ok"


def amotf_core() -> Dict[str, Any]:
    return {
        "stable_argsort": stable_argsort,
        "permutation_to_index": permutation_to_index,
        "get_permutation_index": get_permutation_index,
        "calculate_emd_weight": calculate_emd_weight,
        "build_amotf_single_modality": build_amotf_single_modality,
        "build_amotf_multimodal": build_amotf_multimodal,
        "make_rgb_from_multichan": make_rgb_from_multichan,
    }

# 封装（从csv中构建amotf并保存结果文件）
def build_amotf_images_from_csv(
    *,
    labels_csv: str,
    out_size: int = 0,
    m: int = 5,
    tau: int = 1,
    spans: Optional[List[int]] = None,
    max_points: int = 2000,
    overwrite: bool = False,
    num_workers: int = 0,
) -> int:
    if spans is None:
        spans = [1, 2, 4, 8]
    rows = load_label_rows(labels_csv)
    if not rows:
        LOGGER.error("No rows in labels_csv=%s", labels_csv)
        return 2

    total = len(rows)
    ok = 0
    fail = 0
    reasons: Dict[str, int] = {}

    payloads = [
        (
            str(remap_known_root(r["original_path"])),
            int(out_size),
            int(m),
            int(tau),
            list(spans or []),
            int(max_points),
            bool(overwrite),
        )
        for r in rows
    ]

    if num_workers and num_workers > 0:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=int(num_workers)) as pool:
            for success, reason in _tqdm(pool.imap(_amotf_worker, payloads), total=total, desc="build_amotf"):
                if success:
                    ok += 1
                else:
                    fail += 1
                reasons[reason] = reasons.get(reason, 0) + 1
    else:
        for payload in _tqdm(payloads, total=total, desc="build_amotf"):
            success, reason = _amotf_worker(payload)
            if success:
                ok += 1
            else:
                fail += 1
            reasons[reason] = reasons.get(reason, 0) + 1

    LOGGER.info("AMOTF build done. total=%d ok=%d fail=%d", total, ok, fail)
    for k, v in sorted(reasons.items(), key=lambda x: (-x[1], x[0]))[:20]:
        LOGGER.info("reason_top: %s = %d", k, v)
    return 0

# 新建partial_cycles目录
def _partial_sample_dir(original_sample_dir: str, *, num_splits: int, partial_index: int, partial_phase: str) -> str:
    return os.path.join(
        str(original_sample_dir),
        "partial_cycles",
        f"seg{int(num_splits):02d}",
        str(partial_phase),
        f"p{int(partial_index):02d}",
    )

# 计算partial_cycles目录下每个partial_index的起始索引(倒序从后1/20（对应partial_index=1）到全量（对应partial_index=20）)
def _partial_start_index(n: int, *, num_splits: int, partial_index: int) -> int:
    n = int(n)
    if n <= 1:
        return 0
    denom = int(num_splits)
    if denom <= 0:
        return 0
    i = int(partial_index)
    i = max(1, min(i, denom))
    frac_start = 1.0 - (float(i) / float(denom))
    start = int(math.floor(frac_start * float(n - 1)))
    start = max(0, min(start, n - 1))
    return start

# 构建partial_cycles目录下每个partial_index的amotf并保存结果文件
def build_and_save_amotf_for_partial_cycle(
    original_sample_dir: str,
    partial_sample_dir: str,
    *,
    out_size: int,
    m: int,
    tau: int,
    spans: List[int],
    max_points: int,
    overwrite: bool,
    num_splits: int,
    partial_index: int,
    partial_phase: str,
) -> Tuple[bool, str]:
    curve_csv = os.path.join(original_sample_dir, "curve.csv")
    if not os.path.exists(curve_csv):
        return False, "missing_curve_csv"

    paths = _amotf_output_paths(partial_sample_dir, max_points=int(max_points))
    out_dir = paths["out_dir"]
    c_png = paths["c_png"]
    d_png = paths["d_png"]
    c_npz = paths["c_npz"]
    d_npz = paths["d_npz"]

    if (
        (not overwrite)
        and c_png.exists()
        and d_png.exists()
        and _is_valid_npz(str(c_npz))
        and _is_valid_npz(str(d_npz))
    ):
        return True, "exists"

    curve = read_curve_csv(curve_csv)
    if curve is None:
        return False, "curve_parse_failed"
    c_trip, d_trip, why = split_charge_discharge_with_soc(curve)
    if c_trip is None or d_trip is None:
        return False, f"split_failed:{why}"

    s_c, v_c, i_c = c_trip
    s_d, v_d, i_d = d_trip

    ph = str(partial_phase).strip().lower()
    if ph not in ("charge", "discharge"):
        ph = "charge"

    if ph == "charge":
        start = _partial_start_index(int(len(v_c)), num_splits=int(num_splits), partial_index=int(partial_index))
        s_c = np.asarray(s_c, dtype=float)[start:]
        v_c = np.asarray(v_c, dtype=float)[start:]
        i_c = np.asarray(i_c, dtype=float)[start:]
    else:
        start = _partial_start_index(int(len(v_d)), num_splits=int(num_splits), partial_index=int(partial_index))
        s_d = np.asarray(s_d, dtype=float)[start:]
        v_d = np.asarray(v_d, dtype=float)[start:]
        i_d = np.asarray(i_d, dtype=float)[start:]

    # 采样（横轴 SOC）
    v_c = resample_series_by_x(s_c, v_c, max_points=max_points)
    i_c = resample_series_by_x(s_c, i_c, max_points=max_points)
    v_d = resample_series_by_x(s_d, v_d, max_points=max_points)
    i_d = resample_series_by_x(s_d, i_d, max_points=max_points)

    v_c = _norm_minmax(v_c)
    v_d = _norm_minmax(v_d)
    i_c = _norm_maxabs(i_c)
    i_d = _norm_maxabs(i_d)
    # 根据充电和放电分别构建amotf
    tens_c = build_amotf_multimodal(v_c, i_c, m=m, tau=tau, span_set=spans)
    tens_d = build_amotf_multimodal(v_d, i_d, m=m, tau=tau, span_set=spans)

    _ensure_dir(out_dir)
    np.savez_compressed(str(c_npz), amotf=tens_c.astype(np.float32))
    np.savez_compressed(str(d_npz), amotf=tens_d.astype(np.float32))

    rgb_c = make_rgb_from_multichan(tens_c)
    rgb_d = make_rgb_from_multichan(tens_d)
    save_png_uint8(rgb_c, c_png, out_size=out_size)
    save_png_uint8(rgb_d, d_png, out_size=out_size)

    return True, "ok"

# 封装partial_cycles目录下每个partial_index的amotf构建
def _partial_amotf_worker(payload: Tuple[str, str, int, int, int, List[int], int, bool, int, int, str]) -> Tuple[bool, str]:
    original_sample_dir, partial_sample_dir, out_size, m, tau, spans, max_points, overwrite, num_splits, partial_index, partial_phase = payload
    return build_and_save_amotf_for_partial_cycle(
        original_sample_dir,
        partial_sample_dir,
        out_size=int(out_size),
        m=int(m),
        tau=int(tau),
        spans=list(spans),
        max_points=int(max_points),
        overwrite=bool(overwrite),
        num_splits=int(num_splits),
        partial_index=int(partial_index),
        partial_phase=str(partial_phase),
    )

# 从csv中每个partial_index的amotf构建
def build_partial_amotf_images_from_csv(
    *,
    labels_csv: str,
    out_size: int = 0,
    m: int = 5,
    tau: int = 1,
    spans: Optional[List[int]] = None,
    max_points: int = 2000,
    overwrite: bool = False,
    num_workers: int = 0,
    num_splits: int = 20,
    partial_index: int = 1,
    partial_phase: str = "charge",
) -> int:
    if spans is None:
        spans = [1, 2, 4, 8]
    rows = load_label_rows(labels_csv)
    if not rows:
        LOGGER.error("No rows in labels_csv=%s", labels_csv)
        return 2

    total = len(rows)
    ok = 0
    fail = 0
    reasons: Dict[str, int] = {}

    payloads: List[Tuple[str, str, int, int, int, List[int], int, bool, int, int, str]] = []
    for r in rows:
        op = str(remap_known_root(r["original_path"]))
        pp = _partial_sample_dir(op, num_splits=int(num_splits), partial_index=int(partial_index), partial_phase=str(partial_phase))
        payloads.append(
            (
                op,
                pp,
                int(out_size),
                int(m),
                int(tau),
                list(spans or []),
                int(max_points),
                bool(overwrite),
                int(num_splits),
                int(partial_index),
                str(partial_phase),
            )
        )

    if num_workers and num_workers > 0:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=int(num_workers)) as pool:
            for success, reason in _tqdm(
                pool.imap(_partial_amotf_worker, payloads),
                total=total,
                desc=f"build_partial_amotf_{str(partial_phase)}_p{int(partial_index):02d}",
            ):
                if success:
                    ok += 1
                else:
                    fail += 1
                reasons[reason] = reasons.get(reason, 0) + 1
    else:
        for payload in _tqdm(payloads, total=total, desc=f"build_partial_amotf_{str(partial_phase)}_p{int(partial_index):02d}"):
            success, reason = _partial_amotf_worker(payload)
            if success:
                ok += 1
            else:
                fail += 1
            reasons[reason] = reasons.get(reason, 0) + 1

    LOGGER.info(
        "Partial AMOTF build done. phase=%s partial_index=%d/%d total=%d ok=%d fail=%d",
        str(partial_phase),
        int(partial_index),
        int(num_splits),
        total,
        ok,
        fail,
    )
    for k, v in sorted(reasons.items(), key=lambda x: (-x[1], x[0]))[:20]:
        LOGGER.info("reason_top: %s = %d", k, v)
    return 0


@dataclass
class SampleRecord:
    sample_id: str
    original_path: str
    dataset_name: str
    assigned_class: str
    y: int
    charge_png: str
    discharge_png: str
    amotf_charge_png: str
    amotf_discharge_png: str
    amotf_charge_npz: str
    amotf_discharge_npz: str


def _load_records(labels_csv: str, *, max_points: int = 2000) -> Tuple[List[SampleRecord], Dict[str, int]]:
    rows = load_label_rows(labels_csv)
    dropped: Dict[str, int] = {}
    out: List[SampleRecord] = []
    for r in rows:
        lab = str(r.get("assigned_class", "")).strip()
        if lab not in LABEL_MAP:
            dropped["invalid_label"] = dropped.get("invalid_label", 0) + 1
            continue
        op = remap_known_root(str(r.get("original_path", "")).strip())
        if not op or not os.path.exists(op):
            dropped["missing_original_path"] = dropped.get("missing_original_path", 0) + 1
            continue
        ds = infer_dataset_name_from_original_path(op)
        if ds is None:
            dropped["unknown_dataset"] = dropped.get("unknown_dataset", 0) + 1
            continue
        sid = str(r.get("sample_id", "")).strip() or os.path.basename(op.rstrip("\\/")) or _sha1_short(op)

        paths = _amotf_output_paths(op, max_points=int(max_points))
        out.append(
            SampleRecord(
                sample_id=sid,
                original_path=op,
                dataset_name=ds,
                assigned_class=lab,
                y=int(LABEL_MAP[lab]),
                charge_png=os.path.join(op, "charge.png"),
                discharge_png=os.path.join(op, "discharge.png"),
                amotf_charge_png=str(paths["c_png"]),
                amotf_discharge_png=str(paths["d_png"]),
                amotf_charge_npz=str(paths["c_npz"]),
                amotf_discharge_npz=str(paths["d_npz"]),
            )
        )
    return out, dropped


def _filter_by_input_mode(records: Sequence[SampleRecord], input_mode: str) -> Tuple[List[SampleRecord], Dict[str, int]]:
    kept: List[SampleRecord] = []
    dropped: Dict[str, int] = {}
    for r in records:
        if input_mode == "png":
            ok = _exists(r.charge_png) and _exists(r.discharge_png)
            reason = "missing_charge_or_discharge_png"
        elif input_mode == "amotf":
            ok = _is_valid_npz(r.amotf_charge_npz) and _is_valid_npz(r.amotf_discharge_npz)
            reason = "missing_or_invalid_amotf_charge_or_discharge_npz"
        else:
            ok = _is_valid_npz(r.amotf_charge_npz) and _is_valid_npz(r.amotf_discharge_npz)
            reason = "missing_or_invalid_amotf_charge_or_discharge_npz"
        if ok:
            kept.append(r)
        else:
            dropped[reason] = dropped.get(reason, 0) + 1
    return kept, dropped


def _make_joint_keys(records: Sequence[SampleRecord]) -> np.ndarray:
    return np.array([f"{r.dataset_name}|{r.assigned_class}" for r in records], dtype=object)

# dinov3 convnext tiny，如果有问题，则用convnext tiny
def _build_backbone(hf_model_id: str, hf_local_only: bool, allow_fallback_backbone: bool):
    try:
        from transformers import AutoModel

        bb = AutoModel.from_pretrained(hf_model_id, trust_remote_code=False, local_files_only=bool(hf_local_only))
        bb.eval()
        return bb, f"transformers::{hf_model_id}"
    except Exception as e:
        if not allow_fallback_backbone:
            raise RuntimeError(
                "Failed to load DINOv3 backbone via transformers. "
                "Install transformers or pass --allow_fallback_backbone to use timm convnext_tiny. "
                f"reason={e}"
            )
        LOGGER.warning("DINOv3 transformers backbone unavailable, fallback to timm convnext_tiny. reason=%s", e)
        import timm

        bb = timm.create_model("convnext_tiny", pretrained=True, num_classes=0, global_pool="avg")
        bb.eval()
        return bb, "timm::convnext_tiny"


def _extract_features(backbone, x3):
    import torch

    out = None
    try:
        out = backbone(pixel_values=x3)
    except TypeError:
        out = backbone(x3)

    if torch.is_tensor(out):
        return out
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        return out.last_hidden_state.mean(dim=1)
    if isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
        t = out[0]
        if t.dim() == 4:
            return t.mean(dim=[2, 3])
        if t.dim() == 3:
            return t.mean(dim=1)
        return t
    raise RuntimeError("Unexpected backbone output")


def _parse_lora_targets(s: str) -> List[str]:
    out: List[str] = []
    for x in str(s or "").split(","):
        x = x.strip()
        if x:
            out.append(x)
    return out


def _inject_lora_linear_modules(
    backbone,
    *,
    lora_r: int,
    lora_alpha: float,
    lora_dropout: float,
    lora_targets: str,
) -> Dict[str, Any]:
    import re
    import torch
    import torch.nn as nn

    class LoRALinear(nn.Module):
        def __init__(self, base: nn.Linear, *, r: int, alpha: float, dropout: float):
            super().__init__()
            if not isinstance(base, nn.Linear):
                raise TypeError("base must be nn.Linear")
            self.base = base
            self.r = int(r)
            self.alpha = float(alpha)
            self.scaling = float(alpha) / float(max(1, int(r)))
            self.dropout = nn.Dropout(p=float(dropout)) if float(dropout) > 0 else nn.Identity()
            for p in self.base.parameters():
                p.requires_grad = False
            if int(self.r) > 0:
                self.lora_A = nn.Parameter(torch.empty(int(self.r), int(base.in_features)))
                self.lora_B = nn.Parameter(torch.zeros(int(base.out_features), int(self.r)))
                nn.init.normal_(self.lora_A, mean=0.0, std=0.01)
            else:
                self.lora_A = None
                self.lora_B = None

        def forward(self, x):
            y = self.base(x)
            if self.lora_A is None or self.lora_B is None:
                return y
            x2 = self.dropout(x)
            z = torch.matmul(x2, self.lora_A.t())
            z = torch.matmul(z, self.lora_B.t())
            return y + z * float(self.scaling)

    pats = _parse_lora_targets(lora_targets)
    use_all = (len(pats) == 0) or any(str(p).lower() in ("all", "*") for p in pats)
    regs = [re.compile(p, flags=re.IGNORECASE) for p in pats] if (not use_all) else []

    def _match(name: str) -> bool:
        if use_all:
            return True
        return any(r.search(name) for r in regs)

    replaced: List[str] = []
    for full_name, mod in list(backbone.named_modules()):
        if not full_name:
            continue
        if not isinstance(mod, nn.Linear):
            continue
        if not _match(full_name):
            continue
        parent = backbone
        parts = full_name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        leaf = parts[-1]
        try:
            setattr(parent, leaf, LoRALinear(mod, r=int(lora_r), alpha=float(lora_alpha), dropout=float(lora_dropout)))
            replaced.append(full_name)
        except Exception:
            continue
    return {"replaced": replaced, "num_replaced": int(len(replaced))}


def _collect_linear_target_leaf_names(backbone, *, lora_targets: str) -> Tuple[List[str], Dict[str, Any]]:
    import re
    import torch.nn as nn

    pats = _parse_lora_targets(lora_targets)
    use_all = (len(pats) == 0) or any(str(p).lower() in ("all", "*") for p in pats)
    regs = [re.compile(p, flags=re.IGNORECASE) for p in pats] if (not use_all) else []

    def _match(name: str) -> bool:
        if use_all:
            return True
        return any(r.search(name) for r in regs)

    leaf: List[str] = []
    full: List[str] = []
    for full_name, mod in list(backbone.named_modules()):
        if not full_name:
            continue
        if not isinstance(mod, nn.Linear):
            continue
        if not _match(full_name):
            continue
        full.append(full_name)
        leaf.append(full_name.split(".")[-1])

    uniq_leaf = sorted(list(dict.fromkeys(leaf)))
    return uniq_leaf, {"matched_full": full, "num_matched": int(len(full)), "num_leaf": int(len(uniq_leaf))}


def _inject_lora_peft(
    backbone,
    *,
    lora_r: int,
    lora_alpha: float,
    lora_dropout: float,
    lora_targets: str,
) -> Tuple[Any, Dict[str, Any]]:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except Exception as e:
        raise RuntimeError(f"peft is required for lora_backend=peft: {e}")

    target_modules, info = _collect_linear_target_leaf_names(backbone, lora_targets=str(lora_targets))
    if not target_modules:
        raise RuntimeError(f"peft LoRA: no nn.Linear matched lora_targets={lora_targets!r}")

    cfg = LoraConfig(
        r=int(lora_r),
        lora_alpha=float(lora_alpha),
        lora_dropout=float(lora_dropout),
        bias="none",
        target_modules=target_modules,
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    bb = get_peft_model(backbone, cfg)
    return bb, {"backend": "peft", **info, "target_modules": target_modules}


def _imagenet_norm_module():
    import torch
    import torch.nn as nn

    class ImageNetNormalize(nn.Module):
        def __init__(self):
            super().__init__()
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)

        def forward(self, x):
            return (x - self.mean) / self.std

    return ImageNetNormalize()


class DualImageDataset:
    def __init__(
        self,
        records: Sequence[SampleRecord],
        input_mode: str,
        img_size: int,
        *,
        npz_norm: str = "log1p_global",
        npz_global_max_log: float = 10.0,
    ):
        self.records = list(records)
        self.input_mode = str(input_mode)
        self.img_size = int(img_size)
        self.npz_norm = str(npz_norm or "log1p_global").strip().lower()
        self.npz_global_max_log = float(npz_global_max_log)

        self.input_channels = 3
        if self.input_mode in ("amotf_npz", "amotf"):
            if self.records:
                try:
                    p0 = remap_known_root(self.records[0].amotf_charge_npz)
                    with np.load(p0) as z:
                        a0 = z["amotf"]
                    a0 = np.asarray(a0)
                    if a0.ndim == 3:
                        h0, w0, k0 = int(a0.shape[0]), int(a0.shape[1]), int(a0.shape[2])
                        if (k0 <= 64) and (h0 > 64) and (w0 > 64):
                            self.input_channels = int(k0)
                        elif (h0 <= 64) and (w0 > 64) and (k0 > 64):
                            self.input_channels = int(h0)
                        else:
                            self.input_channels = int(k0)
                except Exception:
                    self.input_channels = 3
            return

        from torchvision import transforms
        from PIL import Image

        self._Image = Image
        self.tfm = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def _load_img(self, p: str):
        with self._Image.open(p) as im:
            return im.convert("RGB").copy()

    def _load_npz_amotf(self, p: str) -> np.ndarray:
        p = remap_known_root(p)
        try:
            with np.load(p, allow_pickle=False) as z:
                a = z["amotf"]
        except Exception as e:
            raise ValueError(f"Failed to load amotf npz: {p} err={e}") from e
        a = np.asarray(a, dtype=np.float32)
        if a.ndim != 3:
            raise RuntimeError(f"Unexpected amotf npz tensor ndim={int(a.ndim)} path={p}")
        h0, w0, k0 = int(a.shape[0]), int(a.shape[1]), int(a.shape[2])
        if (k0 <= 64) and (h0 > 64) and (w0 > 64):
            hwk = a
        elif (h0 <= 64) and (w0 > 64) and (k0 > 64):
            hwk = np.transpose(a, (1, 2, 0))
        else:
            hwk = a
        hwk = np.nan_to_num(hwk, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        mode = str(getattr(self, "npz_norm", "log1p_global") or "log1p_global").strip().lower()

        if mode in ("per_sample_minmax", "sample_minmax", "minmax", "legacy"):
            eps = np.float32(1e-6)
            vmin = np.min(hwk, axis=(0, 1), keepdims=True)
            vmax = np.max(hwk, axis=(0, 1), keepdims=True)
            denom = np.maximum(vmax - vmin, eps)
            hwk = (hwk - vmin) / denom
            hwk = np.clip(hwk, 0.0, 1.0)
        elif mode in ("log1p_global", "log", "log1p"):
            hwk = np.maximum(hwk, np.float32(0.0))
            hwk = np.log1p(hwk)
            denom = float(getattr(self, "npz_global_max_log", 10.0) or 10.0)
            if denom <= 0:
                denom = 10.0
            hwk = hwk / np.float32(denom)
            hwk = np.clip(hwk, 0.0, 1.0)
        else:
            hwk = hwk.astype(np.float32)
        return hwk

    def __getitem__(self, idx: int):
        r = self.records[idx]
        if self.input_mode in ("amotf_npz", "amotf"):
            import torch

            cc = self._load_npz_amotf(r.amotf_charge_npz)
            dd = self._load_npz_amotf(r.amotf_discharge_npz)
            xc = torch.from_numpy(np.transpose(cc, (2, 0, 1))).contiguous()
            xd = torch.from_numpy(np.transpose(dd, (2, 0, 1))).contiguous()
            return (xc, xd), int(r.y)

        if self.input_mode == "png":
            cp, dp = r.charge_png, r.discharge_png
        else:
            cp, dp = r.amotf_charge_png, r.amotf_discharge_png
        xc = self.tfm(self._load_img(cp))
        xd = self.tfm(self._load_img(dp))
        return (xc, xd), int(r.y)


def _build_model(
    backbone,
    *,
    fusion: str,
    finetune_backbone: bool,
    use_lora: bool = False,
    chan_proj: str = "mlp",
    chan_hidden: int = 0,
    chan_norm: str = "group",
    chan_dropout: float = 0.0,
    dropout: float = 0.1,
    img_size: int = 224,
    input_channels: int = 3,
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    for n, p in backbone.named_parameters():
        p.requires_grad = bool(finetune_backbone)
        if bool(use_lora) and (not bool(finetune_backbone)):
            nnm = str(n)
            if ("lora_" in nnm) or (".lora_A" in nnm) or (".lora_B" in nnm):
                p.requires_grad = True

    norm = _imagenet_norm_module()

    in_ch = int(input_channels)
    if in_ch <= 0:
        in_ch = 3

    chan_proj = str(chan_proj or "mlp").strip().lower()
    chan_norm = str(chan_norm or "group").strip().lower()

    def _pick_gn_groups(c: int) -> int:
        c = int(c)
        if c <= 0:
            return 1
        for g in [32, 16, 8, 4, 2, 1]:
            if g <= c and (c % g == 0):
                return int(g)
        return 1

    def _make_norm(c: int) -> nn.Module:
        if str(chan_norm) in ("", "none", "no", "false", "0"):
            return nn.Identity()
        if str(chan_norm) in ("group", "gn", "groupnorm"):
            return nn.GroupNorm(num_groups=_pick_gn_groups(int(c)), num_channels=int(c))
        return nn.Identity()

    if in_ch == 3:
        chan = nn.Identity()
    else:
        if chan_proj in ("linear", "conv", "1x1"):
            chan = nn.Conv2d(in_ch, 3, kernel_size=1, bias=True)
        else:
            hidden = int(chan_hidden)
            if hidden <= 0:
                hidden = int(max(16, min(128, 2 * int(in_ch))))

            class ChanMLP(nn.Module):
                def __init__(self, in_ch: int, hidden: int, out_ch: int = 3):
                    super().__init__()
                    self.norm0 = _make_norm(int(in_ch))
                    self.conv1 = nn.Conv2d(int(in_ch), int(hidden), kernel_size=1, bias=True)
                    self.act = nn.GELU()
                    self.drop = nn.Dropout(p=float(chan_dropout)) if float(chan_dropout) > 0 else nn.Identity()
                    self.norm1 = _make_norm(int(hidden))
                    self.conv2 = nn.Conv2d(int(hidden), int(out_ch), kernel_size=1, bias=True)

                def forward(self, x):
                    x = self.norm0(x)
                    x = self.conv1(x)
                    x = self.act(x)
                    x = self.drop(x)
                    x = self.norm1(x)
                    return self.conv2(x)

            chan = ChanMLP(in_ch=int(in_ch), hidden=int(hidden), out_ch=3)

    class Prep(nn.Module):
        def __init__(self, chan, norm, img_size: int):
            super().__init__()
            self.chan = chan
            self.norm = norm
            self.img_size = int(img_size)

        def forward(self, x):
            x = x.float()
            x = self.chan(x)
            if int(x.shape[-2]) != int(self.img_size) or int(x.shape[-1]) != int(self.img_size):
                x = F.interpolate(x, size=(int(self.img_size), int(self.img_size)), mode="bilinear", align_corners=False)
            return self.norm(x)

    prep = Prep(chan, norm, int(img_size))

    class Feat(nn.Module):
        def __init__(self, bb):
            super().__init__()
            self.bb = bb

        def forward(self, x3):
            return _extract_features(self.bb, x3)

    feat = Feat(backbone)

    with torch.no_grad():
        # Use the same device as the backbone's parameters to avoid CPU/GPU mismatch
        try:
            first_param = next(backbone.parameters())
            bb_dev = first_param.device
        except StopIteration:
            bb_dev = torch.device("cpu")
        dummy = torch.zeros(1, in_ch, img_size, img_size, device=bb_dev)
        prep = prep.to(bb_dev)
        z = feat(prep(dummy))
        fdim = int(z.shape[1])

    in_dim = fdim * 2 if fusion == "concat" else fdim

    head = nn.Linear(in_dim, 3)

    class Cls(nn.Module):
        def __init__(self, feat, head, prep, fusion):
            super().__init__()
            self.feat = feat
            self.head = head
            self.prep = prep
            self.fusion = fusion
            self.finetune_backbone = bool(finetune_backbone)
            self.use_lora = bool(use_lora)

        def train(self, mode: bool = True):
            super().train(mode)
            if not self.finetune_backbone:
                self.feat.bb.eval()
            return self

        def forward(self, x_pair):
            import torch

            xc, xd = x_pair
            xc = self.prep(xc)
            xd = self.prep(xd)
            if self.finetune_backbone or self.use_lora:
                if not self.finetune_backbone:
                    self.feat.bb.eval()
                zc = self.feat(xc)
                zd = self.feat(xd)
            else:
                self.feat.bb.eval()
                with torch.no_grad():
                    zc = self.feat(xc)
                    zd = self.feat(xd)
            if self.fusion == "concat":
                z = torch.cat([zc, zd], dim=1)
            else:
                z = 0.5 * (zc + zd)
            return self.head(z)

    return Cls(feat, head, prep, fusion)


def _eval_probs(model, loader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    import torch

    model.eval()
    ys: List[np.ndarray] = []
    probs: List[np.ndarray] = []
    with torch.no_grad():
        for (xc, xd), y in loader:
            xc = xc.to(device, non_blocking=True)
            xd = xd.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model((xc, xd))
            p = torch.softmax(logits, dim=1).detach().cpu().numpy()
            ys.append(y.detach().cpu().numpy())
            probs.append(p)
    y_true = np.concatenate(ys, axis=0) if ys else np.zeros((0,), dtype=int)
    y_prob = np.concatenate(probs, axis=0) if probs else np.zeros((0, 3), dtype=float)
    return y_true, y_prob


def _metrics_from_probs(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    sk = _try_import_sklearn()
    if sk is None:
        raise RuntimeError("sklearn is required")
    _, _, accuracy_score, confusion_matrix, f1_score = sk

    # Import additional metrics locally
    try:
        from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, roc_auc_score, cohen_kappa_score, average_precision_score
        from sklearn.preprocessing import label_binarize
    except Exception as e:
        raise RuntimeError(f"sklearn metrics import failed: {e}")

    if y_true.size == 0:
        return {
            "acc": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "weighted_precision": 0.0,
            "weighted_recall": 0.0,
            "weighted_f1": 0.0,
            "per_class_precision": [0.0, 0.0, 0.0],
            "per_class_recall": [0.0, 0.0, 0.0],
            "per_class_f1": [0.0, 0.0, 0.0],
            "confusion_matrix": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            "roc_auc_ovr_macro": 0.0,
            "roc_auc_ovr_weighted": 0.0,
            "auprc_macro": 0.0,
            "auprc_weighted": 0.0,
            "cohen_kappa": 0.0,
        }

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = np.argmax(y_prob, axis=1)

    # Basic metrics
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))

    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    weighted_rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))

    pc_prec, pc_rec, pc_f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    pc_prec = [float(x) for x in pc_prec]
    pc_rec = [float(x) for x in pc_rec]
    pc_f1 = [float(x) for x in pc_f1]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).astype(int).tolist()

    # ROC-AUC and AUPRC (macro/weighted) with OVR; guard against degenerate cases
    try:
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
        roc_auc_macro = float(roc_auc_score(y_true_bin, y_prob, multi_class="ovr", average="macro"))
        roc_auc_weighted = float(roc_auc_score(y_true_bin, y_prob, multi_class="ovr", average="weighted"))
    except Exception:
        roc_auc_macro = 0.0
        roc_auc_weighted = 0.0

    try:
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
        auprc_macro = float(average_precision_score(y_true_bin, y_prob, average="macro"))
        auprc_weighted = float(average_precision_score(y_true_bin, y_prob, average="weighted"))
    except Exception:
        auprc_macro = 0.0
        auprc_weighted = 0.0

    try:
        kappa = float(cohen_kappa_score(y_true, y_pred))
    except Exception:
        kappa = 0.0

    return {
        "acc": acc,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_prec,
        "weighted_recall": weighted_rec,
        "weighted_f1": weighted_f1,
        "per_class_precision": pc_prec,
        "per_class_recall": pc_rec,
        "per_class_f1": pc_f1,
        "confusion_matrix": cm,
        "roc_auc_ovr_macro": roc_auc_macro,
        "roc_auc_ovr_weighted": roc_auc_weighted,
        "auprc_macro": auprc_macro,
        "auprc_weighted": auprc_weighted,
        "cohen_kappa": kappa,
    }


def _bad_cases(
    *,
    records: Sequence[SampleRecord],
    indices: Sequence[int],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fold_id: int,
    split: str,
    method: str,
    confidence_gap_threshold: float,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if y_true.size == 0:
        return out
    y_pred = np.argmax(y_prob, axis=1)
    for local_i, (yt, yp) in enumerate(zip(y_true, y_pred)):
        if int(yt) == int(yp):
            continue
        ridx = int(indices[local_i])
        r = records[ridx]
        prob = y_prob[local_i]
        order = np.argsort(prob)[::-1]
        top1 = float(prob[order[0]])
        top2 = float(prob[order[1]]) if len(order) > 1 else 0.0
        gap = float(top1 - top2)
        out.append(
            {
                "sample_id": r.sample_id,
                "original_path": r.original_path,
                "dataset_name": r.dataset_name,
                "assigned_class": r.assigned_class,
                "predicted_class": INV_LABEL.get(int(yp), str(int(yp))),
                "fold_id": int(fold_id),
                "split": str(split),
                "method": str(method),
                "predicted_prob_A": float(prob[0]),
                "predicted_prob_B": float(prob[1]),
                "predicted_prob_C": float(prob[2]),
                "confidence_gap": gap,
                "is_high_confidence_error": bool(gap >= float(confidence_gap_threshold)),
                "charge_image_path": _p_or_missing(r.charge_png),
                "discharge_image_path": _p_or_missing(r.discharge_png),
                "amotf_charge_image_path": _p_or_missing(r.amotf_charge_png),
                "amotf_discharge_image_path": _p_or_missing(r.amotf_discharge_png),
            }
        )
    return out


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    _ensure_dir(path.parent)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    cols: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in cols:
                cols.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _class_counts_from_indices(records: Sequence['SampleRecord'], indices: Sequence[int], num_classes: int = 3) -> Tuple[List[int], List[float], int]:
    ys = np.array([int(records[i].y) for i in indices], dtype=int) if len(indices) else np.zeros((0,), dtype=int)
    cnt = np.bincount(ys, minlength=num_classes).astype(int)
    total = int(cnt.sum())
    ratios = (cnt / max(1, total)).astype(float)
    return cnt.tolist(), ratios.tolist(), total


def _class_counts_from_all(records: Sequence['SampleRecord'], num_classes: int = 3) -> Tuple[List[int], List[float], int]:
    ys = np.array([int(r.y) for r in records], dtype=int) if len(records) else np.zeros((0,), dtype=int)
    cnt = np.bincount(ys, minlength=num_classes).astype(int)
    total = int(cnt.sum())
    ratios = (cnt / max(1, total)).astype(float)
    return cnt.tolist(), ratios.tolist(), total


def _counts_to_label_dict(counts: Sequence[int]) -> Dict[str, int]:
    d: Dict[str, int] = {}
    for i, c in enumerate(counts):
        d[INV_LABEL.get(int(i), str(int(i)))] = int(c)
    return d


def _write_overall_class_distribution_csv(path: Path, counts: Sequence[int], ratios: Sequence[float]) -> None:
    rows: List[Dict[str, Any]] = []
    for i, (c, r) in enumerate(zip(counts, ratios)):
        rows.append({
            "class": INV_LABEL.get(int(i), str(int(i))),
            "count": int(c),
            "ratio": float(r),
        })
    _write_csv(rows, path)


def _write_split_class_distribution_csv(path: Path, split_map: Dict[str, Tuple[Sequence[int], Sequence[float], int]]) -> None:
    # One row per split with counts and ratios for A/B/C
    rows: List[Dict[str, Any]] = []
    for split, (cnt, rat, total) in split_map.items():
        row: Dict[str, Any] = {
            "split": split,
            "total": int(total),
        }
        for i, (c, r) in enumerate(zip(cnt, rat)):
            lab = INV_LABEL.get(int(i), str(int(i)))
            row[f"count_{lab}"] = int(c)
            row[f"ratio_{lab}"] = float(r)
        rows.append(row)
    _write_csv(rows, path)

# 基于dino训练soh分类器
def train_dino_soh_classifier(
    *,
    labels_csv: str,
    runs_root: str,
    run_name: str,
    input_mode: str,
    max_points: int,
    kfold: int,
    split_indices_json: str = "",
    exclude_samples_txt: str = "",
    seed: int,
    img_size: int,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    num_workers: int,
    fusion: str,
    val_ratio: float,
    metric_for_best: str,
    confidence_gap_threshold: float,
    finetune_backbone: bool,
    lr_scheduler: str = "none",
    lr_warmup_ratio: float = 0.0,
    lr_min: float = 0.0,
    lr_plateau_factor: float = 0.5,
    lr_plateau_patience: int = 2,
    backbone_lr_mult: float = 1.0,
    chan_proj: str = "mlp",
    chan_hidden: int = 0,
    chan_norm: str = "group",
    chan_dropout: float = 0.0,
    npz_norm: str = "log1p_global",
    npz_global_max_log: float = 10.0,
    use_lora: bool = False,
    lora_backend: str = "auto",
    lora_r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    lora_targets: str = "pwconv,fc,qkv,proj",
    allow_fallback_backbone: bool,
    hf_model_id: str,
    hf_local_only: bool,
    group_by_sample_id: bool = False,
    use_class_weights: bool = False,
    **_: Any,
) -> int:
    StratifiedKFold, StratifiedShuffleSplit, _, _, _ = _ensure_sklearn()

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Subset

    _set_seed(int(seed))

    split_indices_json = remap_known_root(str(split_indices_json or "").strip())
    exclude_samples_txt = str(exclude_samples_txt or "").strip() or _default_exclude_samples_txt()

    if not run_name:
        run_name = _now_run_name()
    run_dir = Path(runs_root) / run_name / input_mode
    _ensure_dir(run_dir)
    _setup_logging(run_dir / "train.log")
    LOGGER.info("step=init_run run_dir=%s", str(run_dir))
    LOGGER.info("labels_csv=%s", str(labels_csv))
    LOGGER.info("input_mode=%s", str(input_mode))
    LOGGER.info("split_indices_json=%s", str(split_indices_json))
    LOGGER.info("exclude_samples_txt=%s", remap_known_root(str(exclude_samples_txt)))

    LOGGER.info("step=load_records")
    records_all, dropped0 = _load_records(labels_csv, max_points=int(max_points))

    LOGGER.info("step=filter_by_input_mode")
    records, dropped1 = _filter_by_input_mode(records_all, input_mode=input_mode)

    records_before_exclude = list(records)
    excluded_keys: set = set()

    dropped = dict(dropped0)
    for k, v in dropped1.items():
        dropped[k] = dropped.get(k, 0) + v

    ex_sid_base, ex_paths = _load_exclude_sets(exclude_samples_txt, labels_csv=labels_csv)
    if ex_sid_base or ex_paths:
        before = int(len(records))
        kept: List[SampleRecord] = []
        n_ex = 0
        for rr in records:
            sidb = _base_sample_id(str(rr.sample_id))
            op = remap_known_root(str(rr.original_path))
            if (sidb in ex_sid_base) or (op in ex_paths):
                n_ex += 1
                k = op if op else sidb
                if k:
                    excluded_keys.add(str(k))
            else:
                kept.append(rr)
        records = kept
        if n_ex > 0:
            dropped["excluded_by_list"] = dropped.get("excluded_by_list", 0) + int(n_ex)
        LOGGER.info("exclude_samples_txt=%s excluded=%d before=%d after=%d", remap_known_root(exclude_samples_txt), int(n_ex), int(before), int(len(records)))

    if len(records) < 20:
        LOGGER.error("Too few usable samples after filtering: %d", len(records))
        LOGGER.error("Dropped summary: %s", dropped)
        return 2

    LOGGER.info("input_mode=%s samples=%d", input_mode, len(records))
    LOGGER.info("dropped=%s", dropped)

    joint_keys = _make_joint_keys(records)

    kfold_eff = int(1 if split_indices_json else kfold)
    if split_indices_json:
        LOGGER.info("split_mode=external_split split_indices_json=%s", str(split_indices_json))
    else:
        LOGGER.info("split_mode=kfold kfold=%d", int(kfold_eff))

    use_groups = bool(group_by_sample_id)
    base_to_indices: Dict[str, List[int]] = {}
    base_list: List[str] = []
    base_joint_keys = None
    if use_groups:
        for idx, rr in enumerate(records):
            sid = str(rr.sample_id)
            base = re.sub(r"_p\d+$", "", sid)
            if base not in base_to_indices:
                base_to_indices[base] = []
                base_list.append(base)
            base_to_indices[base].append(int(idx))
        base_joint_keys = np.array([joint_keys[base_to_indices[b][0]] for b in base_list], dtype=object)
        uniq, cnt = np.unique(base_joint_keys, return_counts=True)
        min_cnt = int(cnt.min()) if cnt.size else 0
        if int(kfold_eff) > 1 and min_cnt < int(kfold_eff):
            LOGGER.error("Some dataset×class strata have too few samples for kfold. min_count=%d kfold=%d", min_cnt, int(kfold_eff))
            top = sorted([(str(u), int(c)) for u, c in zip(uniq, cnt)], key=lambda x: x[1])[:20]
            LOGGER.error("strata_count_head=%s", top)
            return 2
    else:
        uniq, cnt = np.unique(joint_keys, return_counts=True)
        min_cnt = int(cnt.min()) if cnt.size else 0
        if int(kfold_eff) > 1 and min_cnt < int(kfold_eff):
            LOGGER.error("Some dataset×class strata have too few samples for kfold. min_count=%d kfold=%d", min_cnt, int(kfold_eff))
            top = sorted([(str(u), int(c)) for u, c in zip(uniq, cnt)], key=lambda x: x[1])[:20]
            LOGGER.error("strata_count_head=%s", top)
            return 2

    use_lora = bool(use_lora)
    lora_backend = str(lora_backend or "auto").strip().lower()
    finetune_backbone = bool(finetune_backbone)
    finetune_backbone_effective = bool(finetune_backbone) and (not bool(use_lora))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bb_shared = None
    bb_name = ""
    if (not finetune_backbone_effective) and (not use_lora):
        bb_shared, bb_name = _build_backbone(
            hf_model_id=hf_model_id,
            hf_local_only=hf_local_only,
            allow_fallback_backbone=allow_fallback_backbone,
        )
        bb_shared.eval()

    LOGGER.info("device=%s backbone=%s", device, bb_name if bb_name else "(per-fold)")

    with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "labels_csv": labels_csv,
                "input_mode": input_mode,
                "max_points": int(max_points),
                "kfold": int(1 if split_indices_json else kfold),
                "split_indices_json": str(split_indices_json),
                "exclude_samples_txt": remap_known_root(str(exclude_samples_txt)),
                "seed": int(seed),
                "img_size": int(img_size),
                "batch_size": int(batch_size),
                "epochs": int(epochs),
                "lr": float(lr),
                "weight_decay": float(weight_decay),
                "lr_scheduler": str(lr_scheduler),
                "lr_warmup_ratio": float(lr_warmup_ratio),
                "lr_min": float(lr_min),
                "lr_plateau_factor": float(lr_plateau_factor),
                "lr_plateau_patience": int(lr_plateau_patience),
                "backbone_lr_mult": float(backbone_lr_mult),
                "fusion": fusion,
                "val_ratio": float(val_ratio),
                "metric_for_best": metric_for_best,
                "confidence_gap_threshold": float(confidence_gap_threshold),
                "finetune_backbone": bool(finetune_backbone_effective),
                "chan_proj": str(chan_proj),
                "chan_hidden": int(chan_hidden),
                "chan_norm": str(chan_norm),
                "chan_dropout": float(chan_dropout),
                "npz_norm": str(npz_norm),
                "npz_global_max_log": float(npz_global_max_log),
                "use_lora": bool(use_lora),
                "lora_backend": str(lora_backend),
                "lora_r": int(lora_r),
                "lora_alpha": float(lora_alpha),
                "lora_dropout": float(lora_dropout),
                "lora_targets": str(lora_targets),
                "backbone": bb_name if bb_name else "per-fold",
                "dropped": dropped,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    ds = DualImageDataset(
        records,
        input_mode=input_mode,
        img_size=int(img_size),
        npz_norm=str(npz_norm),
        npz_global_max_log=float(npz_global_max_log),
    )
    ds_in_ch = int(getattr(ds, "input_channels", 3))

    # Write overall class distribution for this method and run
    counts_all, ratios_all, total_all = _class_counts_from_all(records)
    _write_overall_class_distribution_csv(run_dir / "class_distribution_overall.csv", counts_all, ratios_all)
    ratios_map_all = {INV_LABEL.get(i, str(i)): float(ratios_all[i]) for i in range(len(ratios_all))}
    LOGGER.info(
        "overall_class_distribution total=%d counts=%s ratios=%s",
        total_all,
        _counts_to_label_dict(counts_all),
        ratios_map_all,
    )

    fold_rows: List[Dict[str, Any]] = []
    all_bad_cases: List[Dict[str, Any]] = []

    ext_tr_idx: Optional[np.ndarray] = None
    ext_va_idx: Optional[np.ndarray] = None
    ext_te_idx: Optional[np.ndarray] = None
    ext_fold_id: int = 0

    if split_indices_json:
        if not os.path.exists(split_indices_json):
            LOGGER.error("split_indices_json not found: %s", split_indices_json)
            return 2
        with open(split_indices_json, "r", encoding="utf-8") as f:
            d = json.load(f)
        try:
            ext_tr_idx = np.array(d.get("train_idx") or [], dtype=int)
            ext_va_idx = np.array(d.get("val_idx") or [], dtype=int)
            ext_te_idx = np.array(d.get("test_idx") or [], dtype=int)
        except Exception as e:
            LOGGER.error("Invalid split_indices_json format: %s err=%s", split_indices_json, e)
            return 2

        all_idx = ext_tr_idx.tolist() + ext_va_idx.tolist() + ext_te_idx.tolist()
        max_idx = int(max(all_idx)) if all_idx else -1
        if max_idx >= int(len(records_before_exclude)):
            LOGGER.error(
                "split_indices_json indices out of range. max_idx=%d n_records(before_exclude)=%d path=%s",
                int(max_idx),
                int(len(records_before_exclude)),
                split_indices_json,
            )
            return 2

        def _stable_key(rr: SampleRecord) -> str:
            op0 = remap_known_root(str(rr.original_path))
            if op0:
                return str(op0)
            return str(_base_sample_id(str(rr.sample_id)))

        key_to_new: Dict[str, int] = {}
        dup_keys: List[str] = []
        for i, rr in enumerate(records):
            kk = _stable_key(rr)
            if not kk:
                continue
            if kk in key_to_new:
                if len(dup_keys) < 20:
                    dup_keys.append(kk)
                continue
            key_to_new[kk] = int(i)
        if dup_keys:
            LOGGER.error("Duplicate stable keys in kept records (showing up to 20): %s", dup_keys)
            return 2

        def _remap_split(old_idx: np.ndarray, split_name: str) -> np.ndarray:
            old_list = [int(x) for x in old_idx.tolist()]
            keys = [_stable_key(records_before_exclude[int(i)]) for i in old_list]
            new_idx_list: List[int] = []
            n_drop_ex = 0
            missing: List[str] = []
            for kk in keys:
                if not kk:
                    continue
                j = key_to_new.get(str(kk))
                if j is not None:
                    new_idx_list.append(int(j))
                else:
                    if str(kk) in excluded_keys:
                        n_drop_ex += 1
                    else:
                        if len(missing) < 20:
                            missing.append(str(kk))
            LOGGER.info(
                "external_split_remap split=%s old_n=%d dropped_excluded=%d new_n=%d",
                str(split_name),
                int(len(old_list)),
                int(n_drop_ex),
                int(len(new_idx_list)),
            )
            if missing:
                LOGGER.error(
                    "external_split_remap missing keys (not excluded) split=%s n_missing=%d examples=%s",
                    str(split_name),
                    int(len(missing)),
                    missing,
                )
                raise RuntimeError("external_split_remap missing keys")
            return np.array(new_idx_list, dtype=int)

        try:
            ext_tr_idx = _remap_split(ext_tr_idx, "train")
            ext_va_idx = _remap_split(ext_va_idx, "val")
            ext_te_idx = _remap_split(ext_te_idx, "test")
        except Exception:
            return 2

        LOGGER.info(
            "external_split_remap_done n_records_before_exclude=%d n_records_after_exclude=%d",
            int(len(records_before_exclude)),
            int(len(records)),
        )

        m = re.search(r"[\\/]fold_(\d+)[\\/]split_indices\.json$", str(split_indices_json))
        ext_fold_id = int(m.group(1)) if m else 0

        split_iter = [(np.array([], dtype=int), np.array([], dtype=int))]
    else:
        kf = StratifiedKFold(n_splits=int(kfold_eff), shuffle=True, random_state=int(seed))
        if use_groups:
            assert base_joint_keys is not None
            split_iter = kf.split(np.zeros(len(base_list)), base_joint_keys)
        else:
            split_iter = kf.split(np.zeros(len(records)), joint_keys)

    for fold_id, (trainval_a, test_a) in enumerate(split_iter):
        if split_indices_json:
            assert ext_tr_idx is not None and ext_va_idx is not None and ext_te_idx is not None
            fold_id = int(ext_fold_id)
            tr_idx = np.array(ext_tr_idx, dtype=int)
            va_idx = np.array(ext_va_idx, dtype=int)
            test_idx = np.array(ext_te_idx, dtype=int)
        elif use_groups:
            trainval_bi = np.array(trainval_a, dtype=int)
            test_bi = np.array(test_a, dtype=int)

            sss = StratifiedShuffleSplit(n_splits=1, test_size=float(val_ratio), random_state=int(seed) + 1000 + int(fold_id))
            tr_bsub, va_bsub = next(sss.split(np.zeros(len(trainval_bi)), base_joint_keys[trainval_bi]))

            tr_idx = np.array([j for bi in trainval_bi[tr_bsub] for j in base_to_indices[base_list[int(bi)]]], dtype=int)
            va_idx = np.array([j for bi in trainval_bi[va_bsub] for j in base_to_indices[base_list[int(bi)]]], dtype=int)
            test_idx = np.array([j for bi in test_bi for j in base_to_indices[base_list[int(bi)]]], dtype=int)
        else:
            trainval_idx = np.array(trainval_a, dtype=int)
            test_idx = np.array(test_a, dtype=int)

            sss = StratifiedShuffleSplit(n_splits=1, test_size=float(val_ratio), random_state=int(seed) + 1000 + int(fold_id))
            tr_sub, va_sub = next(sss.split(np.zeros(len(trainval_idx)), joint_keys[trainval_idx]))
            tr_idx = trainval_idx[tr_sub]
            va_idx = trainval_idx[va_sub]

        fold_dir = run_dir / f"fold_{fold_id}"
        _ensure_dir(fold_dir)
        with open(fold_dir / "split_indices.json", "w", encoding="utf-8") as f:
            json.dump(
                {"train_idx": tr_idx.tolist(), "val_idx": va_idx.tolist(), "test_idx": test_idx.tolist()},
                f,
                ensure_ascii=False,
                indent=2,
            )

        # Per-split class distributions for this fold
        cnt_tr, rat_tr, tot_tr = _class_counts_from_indices(records, tr_idx.tolist())
        cnt_va, rat_va, tot_va = _class_counts_from_indices(records, va_idx.tolist())
        cnt_te, rat_te, tot_te = _class_counts_from_indices(records, test_idx.tolist())
        _write_split_class_distribution_csv(
            fold_dir / "class_distribution_splits.csv",
            {
                "train": (cnt_tr, rat_tr, tot_tr),
                "val": (cnt_va, rat_va, tot_va),
                "test": (cnt_te, rat_te, tot_te),
            },
        )
        LOGGER.info(
            "fold=%d split_distributions train=%s val=%s test=%s",
            fold_id,
            {"total": tot_tr, "counts": _counts_to_label_dict(cnt_tr), "ratios": {INV_LABEL.get(i, str(i)): float(rat_tr[i]) for i in range(len(rat_tr))}},
            {"total": tot_va, "counts": _counts_to_label_dict(cnt_va), "ratios": {INV_LABEL.get(i, str(i)): float(rat_va[i]) for i in range(len(rat_va))}},
            {"total": tot_te, "counts": _counts_to_label_dict(cnt_te), "ratios": {INV_LABEL.get(i, str(i)): float(rat_te[i]) for i in range(len(rat_te))}},
        )

        train_loader = DataLoader(Subset(ds, tr_idx.tolist()), batch_size=int(batch_size), shuffle=True, num_workers=int(num_workers), pin_memory=True)
        val_loader = DataLoader(Subset(ds, va_idx.tolist()), batch_size=int(batch_size), shuffle=False, num_workers=int(num_workers), pin_memory=True)
        test_loader = DataLoader(Subset(ds, test_idx.tolist()), batch_size=int(batch_size), shuffle=False, num_workers=int(num_workers), pin_memory=True)

        if finetune_backbone_effective or use_lora:
            bb, bb_name_local = _build_backbone(
                hf_model_id=hf_model_id,
                hf_local_only=hf_local_only,
                allow_fallback_backbone=allow_fallback_backbone,
            )
            bb.eval()
            if use_lora:
                backend = str(lora_backend)
                if backend == "auto":
                    try:
                        bb, info = _inject_lora_peft(
                            bb,
                            lora_r=int(lora_r),
                            lora_alpha=float(lora_alpha),
                            lora_dropout=float(lora_dropout),
                            lora_targets=str(lora_targets),
                        )
                        bb.eval()
                    except Exception:
                        info = _inject_lora_linear_modules(
                            bb,
                            lora_r=int(lora_r),
                            lora_alpha=float(lora_alpha),
                            lora_dropout=float(lora_dropout),
                            lora_targets=str(lora_targets),
                        )
                elif backend == "peft":
                    bb, info = _inject_lora_peft(
                        bb,
                        lora_r=int(lora_r),
                        lora_alpha=float(lora_alpha),
                        lora_dropout=float(lora_dropout),
                        lora_targets=str(lora_targets),
                    )
                    bb.eval()
                else:
                    info = _inject_lora_linear_modules(
                        bb,
                        lora_r=int(lora_r),
                        lora_alpha=float(lora_alpha),
                        lora_dropout=float(lora_dropout),
                        lora_targets=str(lora_targets),
                    )
                try:
                    nlin = int(info.get("num_replaced", info.get("num_matched", 0)))
                    LOGGER.info("lora_injected fold=%d backend=%s num_linear=%d", int(fold_id), str(info.get("backend", lora_backend)), int(nlin))
                except Exception:
                    pass
        else:
            bb = bb_shared
            bb_name_local = bb_name

        model = _build_model(
            bb,
            fusion=fusion,
            finetune_backbone=bool(finetune_backbone_effective),
            use_lora=bool(use_lora),
            chan_proj=str(chan_proj),
            chan_hidden=int(chan_hidden),
            chan_norm=str(chan_norm),
            chan_dropout=float(chan_dropout),
            img_size=int(img_size),
            input_channels=ds_in_ch,
        )
        model.to(device)

        base_lr = float(lr)
        bb_mult = float(backbone_lr_mult)
        if bb_mult <= 0:
            bb_mult = 1.0

        if bool(finetune_backbone_effective) and float(bb_mult) != 1.0:
            bb_params = []
            other_params = []
            for n, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if str(n).startswith("feat.bb."):
                    bb_params.append(p)
                else:
                    other_params.append(p)
            param_groups = []
            if other_params:
                param_groups.append({"params": other_params, "lr": float(base_lr)})
            if bb_params:
                param_groups.append({"params": bb_params, "lr": float(base_lr) * float(bb_mult)})
            opt = optim.AdamW(param_groups, lr=float(base_lr), weight_decay=float(weight_decay))
        else:
            params = [p for p in model.parameters() if p.requires_grad]
            opt = optim.AdamW(params, lr=float(base_lr), weight_decay=float(weight_decay))

        sched_name = str(lr_scheduler or "none").strip().lower()
        sched = None
        sched_step_per_batch = False
        steps_per_epoch = max(1, int(len(train_loader)))
        total_steps = int(epochs) * int(steps_per_epoch)
        warmup_ratio = float(lr_warmup_ratio)
        if warmup_ratio < 0:
            warmup_ratio = 0.0
        if warmup_ratio > 0.9:
            warmup_ratio = 0.9
        eta_min = float(lr_min)
        if eta_min < 0:
            eta_min = 0.0

        if sched_name in ("", "none", "off", "false", "0"):
            sched = None
        elif sched_name in ("cosine", "cos"):
            sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(max(1, int(epochs))), eta_min=float(eta_min))
            sched_step_per_batch = False
        elif sched_name in ("cosine_warmup", "warmup_cosine", "cosinewarmup"):
            warmup_steps = int(round(float(total_steps) * float(warmup_ratio)))
            if warmup_steps < 0:
                warmup_steps = 0
            if warmup_steps >= int(total_steps):
                warmup_steps = int(max(0, int(total_steps) - 1))
            denom = float(max(1, int(total_steps) - int(warmup_steps)))
            base = float(base_lr)
            min_lr_local = float(eta_min)

            def _lr_mult(step: int) -> float:
                s = int(step) + 1
                if int(warmup_steps) > 0 and s <= int(warmup_steps):
                    return float(s) / float(max(1, int(warmup_steps)))
                if int(total_steps) <= 1:
                    return 1.0
                prog = float(s - int(warmup_steps)) / float(denom)
                prog = float(min(1.0, max(0.0, prog)))
                target = float(min_lr_local) + 0.5 * (float(base) - float(min_lr_local)) * (1.0 + math.cos(math.pi * float(prog)))
                if float(base) <= 0:
                    return 1.0
                return float(target) / float(base)

            sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_mult)
            sched_step_per_batch = True
        elif sched_name in ("onecycle", "one_cycle", "1cycle"):
            max_lrs = [float(pg.get("lr", base_lr)) for pg in opt.param_groups]
            sched = optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=max_lrs,
                total_steps=int(max(1, int(total_steps))),
                pct_start=float(max(0.0, min(0.9, warmup_ratio if warmup_ratio > 0 else 0.1))),
                anneal_strategy="cos",
                div_factor=25.0,
                final_div_factor=1.0e4,
            )
            sched_step_per_batch = True
        elif sched_name in ("plateau", "reduce", "reducelronplateau"):
            sched = optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                mode="max",
                factor=float(lr_plateau_factor),
                patience=int(lr_plateau_patience),
                min_lr=float(eta_min),
            )
            sched_step_per_batch = False
        else:
            raise ValueError(f"Unknown lr_scheduler={lr_scheduler!r}. Choose from none/cosine/cosine_warmup/onecycle/plateau")

        crit = None
        if bool(use_class_weights):
            cnt_for_w, _, _ = _class_counts_from_indices(records, tr_idx.tolist())
            w = np.asarray(cnt_for_w, dtype=np.float32)
            w = 1.0 / np.maximum(w, 1.0)
            w = w / np.maximum(w.sum(), 1e-12)
            crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
        else:
            crit = nn.CrossEntropyLoss()

        best_score = -1e9
        best_state = None
        best_epoch = -1

        for ep in range(int(epochs)):
            model.train(True)
            loss_sum = 0.0
            n = 0
            for (xc, xd), yy in train_loader:
                xc = xc.to(device, non_blocking=True)
                xd = xd.to(device, non_blocking=True)
                yy = yy.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                logits = model((xc, xd))
                loss = crit(logits, yy)
                loss.backward()
                opt.step()
                if sched is not None and bool(sched_step_per_batch):
                    sched.step()
                bsz = int(yy.size(0))
                loss_sum += float(loss.item()) * bsz
                n += bsz

            yv_true, yv_prob = _eval_probs(model, val_loader, device)
            mv = _metrics_from_probs(yv_true, yv_prob)
            score = float(mv.get(metric_for_best, 0.0))

            if sched is not None and (not bool(sched_step_per_batch)):
                if isinstance(sched, optim.lr_scheduler.ReduceLROnPlateau):
                    sched.step(float(score))
                else:
                    sched.step()

            lrs = [float(pg.get("lr", 0.0)) for pg in opt.param_groups]
            lr_min_cur = float(min(lrs)) if lrs else 0.0
            lr_max_cur = float(max(lrs)) if lrs else 0.0

            LOGGER.info(
                "fold=%d epoch=%d train_loss=%.6f val_acc=%.4f val_macro_f1=%.4f lr=[%.3e,%.3e]",
                fold_id,
                ep,
                (loss_sum / max(1, n)),
                float(mv["acc"]),
                float(mv["macro_f1"]),
                float(lr_min_cur),
                float(lr_max_cur),
            )

            if score > best_score:
                best_score = score
                best_epoch = ep
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        yv_true, yv_prob = _eval_probs(model, val_loader, device)
        yt_true, yt_prob = _eval_probs(model, test_loader, device)
        mv = _metrics_from_probs(yv_true, yv_prob)
        mt = _metrics_from_probs(yt_true, yt_prob)

        torch.save({"model": model.state_dict(), "best_epoch": int(best_epoch), "best_score": float(best_score)}, fold_dir / "best.pt")
        # 保存错误的case
        val_bad = _bad_cases(
            records=records,
            indices=va_idx.tolist(),
            y_true=yv_true,
            y_prob=yv_prob,
            fold_id=fold_id,
            split="val",
            method=input_mode,
            confidence_gap_threshold=float(confidence_gap_threshold),
        )
        test_bad = _bad_cases(
            records=records,
            indices=test_idx.tolist(),
            y_true=yt_true,
            y_prob=yt_prob,
            fold_id=fold_id,
            split="test",
            method=input_mode,
            confidence_gap_threshold=float(confidence_gap_threshold),
        )
        _write_csv(val_bad, fold_dir / "bad_cases_val.csv")
        _write_csv(test_bad, fold_dir / "bad_cases_test.csv")
        all_bad_cases.extend(val_bad)
        all_bad_cases.extend(test_bad)

        fold_rows.append(
            {
                "fold": int(fold_id),
                "method": input_mode,
                "n_train": int(len(tr_idx)),
                "n_val": int(len(va_idx)),
                "n_test": int(len(test_idx)),
                "best_epoch": int(best_epoch),
                "best_score": float(best_score),
                # Validation metrics
                "val_acc": float(mv["acc"]),
                "val_macro_precision": float(mv.get("macro_precision", 0.0)),
                "val_macro_recall": float(mv.get("macro_recall", 0.0)),
                "val_macro_f1": float(mv.get("macro_f1", 0.0)),
                "val_weighted_precision": float(mv.get("weighted_precision", 0.0)),
                "val_weighted_recall": float(mv.get("weighted_recall", 0.0)),
                "val_weighted_f1": float(mv.get("weighted_f1", 0.0)),
                "val_roc_auc_ovr_macro": float(mv.get("roc_auc_ovr_macro", 0.0)),
                "val_roc_auc_ovr_weighted": float(mv.get("roc_auc_ovr_weighted", 0.0)),
                "val_auprc_macro": float(mv.get("auprc_macro", 0.0)),
                "val_auprc_weighted": float(mv.get("auprc_weighted", 0.0)),
                "val_cohen_kappa": float(mv.get("cohen_kappa", 0.0)),
                "val_per_class_precision": json.dumps(mv.get("per_class_precision", [])),
                "val_per_class_recall": json.dumps(mv.get("per_class_recall", [])),
                "val_per_class_f1": json.dumps(mv.get("per_class_f1", [])),
                "val_confusion_matrix": json.dumps(mv.get("confusion_matrix", [])),
                # Test metrics
                "test_acc": float(mt["acc"]),
                "test_macro_precision": float(mt.get("macro_precision", 0.0)),
                "test_macro_recall": float(mt.get("macro_recall", 0.0)),
                "test_macro_f1": float(mt.get("macro_f1", 0.0)),
                "test_weighted_precision": float(mt.get("weighted_precision", 0.0)),
                "test_weighted_recall": float(mt.get("weighted_recall", 0.0)),
                "test_weighted_f1": float(mt.get("weighted_f1", 0.0)),
                "test_roc_auc_ovr_macro": float(mt.get("roc_auc_ovr_macro", 0.0)),
                "test_roc_auc_ovr_weighted": float(mt.get("roc_auc_ovr_weighted", 0.0)),
                "test_auprc_macro": float(mt.get("auprc_macro", 0.0)),
                "test_auprc_weighted": float(mt.get("auprc_weighted", 0.0)),
                "test_cohen_kappa": float(mt.get("cohen_kappa", 0.0)),
                "test_per_class_precision": json.dumps(mt.get("per_class_precision", [])),
                "test_per_class_recall": json.dumps(mt.get("per_class_recall", [])),
                "test_per_class_f1": json.dumps(mt.get("per_class_f1", [])),
                "test_confusion_matrix": json.dumps(mt.get("confusion_matrix", [])),
            }
        )

    _write_csv(fold_rows, run_dir / "fold_metrics.csv")
    _write_csv(all_bad_cases, run_dir / "bad_cases_all.csv")

    def _agg(key: str) -> Tuple[float, float]:
        vals = [float(r[key]) for r in fold_rows]
        return float(np.mean(vals)), float(np.std(vals))

    summary = {
        "method": input_mode,
        "kfold": int(1 if split_indices_json else kfold),
        "test_acc_mean": _agg("test_acc")[0],
        "test_acc_std": _agg("test_acc")[1],
        "test_macro_precision_mean": _agg("test_macro_precision")[0],
        "test_macro_precision_std": _agg("test_macro_precision")[1],
        "test_macro_recall_mean": _agg("test_macro_recall")[0],
        "test_macro_recall_std": _agg("test_macro_recall")[1],
        "test_macro_f1_mean": _agg("test_macro_f1")[0],
        "test_macro_f1_std": _agg("test_macro_f1")[1],
        "test_weighted_precision_mean": _agg("test_weighted_precision")[0],
        "test_weighted_precision_std": _agg("test_weighted_precision")[1],
        "test_weighted_recall_mean": _agg("test_weighted_recall")[0],
        "test_weighted_recall_std": _agg("test_weighted_recall")[1],
        "test_weighted_f1_mean": _agg("test_weighted_f1")[0],
        "test_weighted_f1_std": _agg("test_weighted_f1")[1],
        "test_roc_auc_ovr_macro_mean": _agg("test_roc_auc_ovr_macro")[0],
        "test_roc_auc_ovr_macro_std": _agg("test_roc_auc_ovr_macro")[1],
        "test_roc_auc_ovr_weighted_mean": _agg("test_roc_auc_ovr_weighted")[0],
        "test_roc_auc_ovr_weighted_std": _agg("test_roc_auc_ovr_weighted")[1],
        "test_auprc_macro_mean": _agg("test_auprc_macro")[0],
        "test_auprc_macro_std": _agg("test_auprc_macro")[1],
        "test_auprc_weighted_mean": _agg("test_auprc_weighted")[0],
        "test_auprc_weighted_std": _agg("test_auprc_weighted")[1],
        "test_cohen_kappa_mean": _agg("test_cohen_kappa")[0],
        "test_cohen_kappa_std": _agg("test_cohen_kappa")[1],
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    LOGGER.info("Done. fold_metrics=%s", str(run_dir / "fold_metrics.csv"))
    LOGGER.info("Done. bad_cases_all=%s", str(run_dir / "bad_cases_all.csv"))
    LOGGER.info("Summary: %s", summary)
    return 0


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    pd = _try_import_pandas()
    if pd is not None:
        try:
            df = pd.read_csv(str(path))
            return df.to_dict(orient="records")
        except Exception:
            pass
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _sum_confusion_from_fold_metrics(fold_metrics_csv: Path) -> List[List[int]]:
    rows = _read_csv_rows(fold_metrics_csv)
    total: Optional[np.ndarray] = None
    for r in rows:
        s = r.get("test_confusion_matrix", "")
        if not s:
            continue
        try:
            cm = np.asarray(json.loads(str(s)), dtype=int)
        except Exception:
            continue
        if cm.ndim != 2:
            continue
        if total is None:
            total = np.zeros_like(cm, dtype=int)
        if total.shape != cm.shape:
            continue
        total += cm
    if total is None:
        return []
    return total.astype(int).tolist()


def _safe_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        s = str(x).strip()
        if s == "" or s.lower() in ("nan", "none"):
            return float("nan")
        return float(s)
    except Exception:
        return float("nan")


def _write_partial_labels_csv(
    in_labels_csv: str,
    *,
    out_csv: Path,
    num_splits: int,
    partial_index: int,
    partial_phase: str,
    max_points: int,
    input_mode: str,
) -> Dict[str, int]:
    rows = load_label_rows(in_labels_csv)
    keep_rows: List[Dict[str, Any]] = []
    stats: Dict[str, int] = {
        "total": 0,
        "keep": 0,
        "missing_partial_dir": 0,
        "missing_input": 0,
    }
    for r in rows:
        op = remap_known_root(str(r.get("original_path", "")).strip())
        if not op:
            continue
        stats["total"] += 1
        pp = _partial_sample_dir(op, num_splits=int(num_splits), partial_index=int(partial_index), partial_phase=str(partial_phase))
        if not os.path.exists(pp):
            stats["missing_partial_dir"] += 1
            continue

        paths = _amotf_output_paths(pp, max_points=int(max_points))
        if str(input_mode) == "png":
            ok = _exists(os.path.join(pp, "charge.png")) and _exists(os.path.join(pp, "discharge.png"))
        else:
            ok = _is_valid_npz(str(paths["c_npz"])) and _is_valid_npz(str(paths["d_npz"]))

        if not ok:
            stats["missing_input"] += 1
            continue

        sid0 = str(r.get("sample_id", "")).strip() or os.path.basename(op.rstrip("\\/")) or _sha1_short(op)
        keep_rows.append(
            {
                "sample_id": f"{sid0}_p{int(partial_index):02d}",
                "original_path": str(pp),
                "assigned_class": str(r.get("assigned_class", "")).strip(),
            }
        )
        stats["keep"] += 1

    _write_csv(keep_rows, out_csv)
    return stats


def _write_partial_labels_ge_csv(
    in_labels_csv: str,
    *,
    out_csv: Path,
    num_splits: int,
    partial_index_min: int,
    partial_phase: str,
    max_points: int,
    input_mode: str,
) -> Dict[str, int]:
    rows = load_label_rows(in_labels_csv)
    keep_rows: List[Dict[str, Any]] = []
    stats: Dict[str, int] = {
        "total": 0,
        "keep": 0,
        "missing_partial_dir": 0,
        "missing_input": 0,
    }
    pmin = int(partial_index_min)
    for r in rows:
        op = remap_known_root(str(r.get("original_path", "")).strip())
        if not op:
            continue
        sid0 = str(r.get("sample_id", "")).strip() or os.path.basename(op.rstrip("\\/")) or _sha1_short(op)
        lab = str(r.get("assigned_class", "")).strip()

        for pidx in range(max(1, pmin), int(num_splits) + 1):
            stats["total"] += 1
            pp = _partial_sample_dir(op, num_splits=int(num_splits), partial_index=int(pidx), partial_phase=str(partial_phase))
            if not os.path.exists(pp):
                stats["missing_partial_dir"] += 1
                continue

            paths = _amotf_output_paths(pp, max_points=int(max_points))
            if str(input_mode) == "png":
                ok = _exists(os.path.join(pp, "charge.png")) and _exists(os.path.join(pp, "discharge.png"))
            else:
                ok = _is_valid_npz(str(paths["c_npz"])) and _is_valid_npz(str(paths["d_npz"]))

            if not ok:
                stats["missing_input"] += 1
                continue

            keep_rows.append(
                {
                    "sample_id": f"{sid0}_p{int(pidx):02d}",
                    "original_path": str(pp),
                    "assigned_class": lab,
                }
            )
            stats["keep"] += 1

    _write_csv(keep_rows, out_csv)
    return stats


def run_partial_sweep(
    *,
    labels_csv: str,
    runs_root: str,
    run_name: str,
    out_size: int,
    m: int,
    tau: int,
    spans: List[int],
    max_points: int,
    overwrite: bool,
    num_workers_build: int,
    num_splits: int,
    partial_phase: str,
    train_input_mode: str,
    pooled_thresholds: List[int],
    partial_mode: str,
    kfold: int,
    split_indices_json: str = "",
    exclude_samples_txt: str = "",
    seed: int,
    img_size: int,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    lr_scheduler: str = "none",
    lr_warmup_ratio: float = 0.0,
    lr_min: float = 0.0,
    lr_plateau_factor: float = 0.5,
    lr_plateau_patience: int = 2,
    backbone_lr_mult: float = 1.0,
    num_workers: int,
    fusion: str,
    val_ratio: float,
    metric_for_best: str,
    confidence_gap_threshold: float,
    finetune_backbone: bool,
    chan_proj: str,
    chan_hidden: int,
    chan_norm: str,
    chan_dropout: float,
    npz_norm: str,
    npz_global_max_log: float,
    use_lora: bool,
    use_class_weights: bool,
    lora_backend: str,
    lora_r: int,
    lora_alpha: float,
    lora_dropout: float,
    lora_targets: str,
    allow_fallback_backbone: bool,
    hf_model_id: str,
    hf_local_only: bool,
) -> int:
    base = str(run_name).strip() or _now_run_name()
    root = Path(runs_root) / base
    _ensure_dir(root)
    _setup_logging(root / "partial_sweep.log")

    LOGGER.info("partial_sweep run_name=%s num_splits=%d phase=%s", base, int(num_splits), str(partial_phase))

    pmode = str(partial_mode or "pooled").strip().lower()
    do_per_partial = pmode in ("per_partial", "per", "a", "both", "all")
    do_pooled = pmode in ("pooled", "b", "both", "all")

    LOGGER.info("step=build_full_amotf")
    rc0 = build_amotf_images_from_csv(
        labels_csv=str(labels_csv),
        out_size=int(out_size),
        m=int(m),
        tau=int(tau),
        spans=list(spans),
        max_points=int(max_points),
        overwrite=bool(overwrite),
        num_workers=int(num_workers_build),
    )
    if rc0 != 0:
        return int(rc0)

    LOGGER.info("step=train_full_baseline")
    rc_full = train_dino_soh_classifier(
        labels_csv=str(labels_csv),
        runs_root=str(runs_root),
        run_name=f"{base}/full",
        input_mode=str(train_input_mode),
        max_points=int(max_points),
        kfold=int(kfold),
        split_indices_json=str(split_indices_json),
        exclude_samples_txt=str(exclude_samples_txt),
        seed=int(seed),
        img_size=int(img_size),
        batch_size=int(batch_size),
        epochs=int(epochs),
        lr=float(lr),
        weight_decay=float(weight_decay),
        lr_scheduler=str(lr_scheduler),
        lr_warmup_ratio=float(lr_warmup_ratio),
        lr_min=float(lr_min),
        lr_plateau_factor=float(lr_plateau_factor),
        lr_plateau_patience=int(lr_plateau_patience),
        backbone_lr_mult=float(backbone_lr_mult),
        num_workers=int(num_workers),
        fusion=str(fusion),
        val_ratio=float(val_ratio),
        metric_for_best=str(metric_for_best),
        confidence_gap_threshold=float(confidence_gap_threshold),
        finetune_backbone=bool(finetune_backbone),
        chan_proj=str(chan_proj),
        chan_hidden=int(chan_hidden),
        chan_norm=str(chan_norm),
        chan_dropout=float(chan_dropout),
        npz_norm=str(npz_norm),
        npz_global_max_log=float(npz_global_max_log),
        use_lora=bool(use_lora),
        use_class_weights=bool(use_class_weights),
        lora_backend=str(lora_backend),
        lora_r=int(lora_r),
        lora_alpha=float(lora_alpha),
        lora_dropout=float(lora_dropout),
        lora_targets=str(lora_targets),
        allow_fallback_backbone=bool(allow_fallback_backbone),
        hf_model_id=str(hf_model_id),
        hf_local_only=bool(hf_local_only),
    )
    if rc_full != 0:
        return int(rc_full)

    full_dir = Path(runs_root) / base / "full" / str(train_input_mode)
    full_sum: Dict[str, Any] = {}
    if (full_dir / "summary.json").exists():
        with open(full_dir / "summary.json", "r", encoding="utf-8") as f:
            full_sum = json.load(f)
    full_cm = _sum_confusion_from_fold_metrics(full_dir / "fold_metrics.csv")

    sweep_rows: List[Dict[str, Any]] = []
    sweep_rows.append(
        {
            "partial_index": int(num_splits),
            "partial_ratio": 1.0,
            "variant": "full",
            "n_samples": "",
            "test_acc_mean": full_sum.get("test_acc_mean", ""),
            "test_acc_std": full_sum.get("test_acc_std", ""),
            "test_macro_f1_mean": full_sum.get("test_macro_f1_mean", ""),
            "test_macro_f1_std": full_sum.get("test_macro_f1_std", ""),
            "test_weighted_f1_mean": full_sum.get("test_weighted_f1_mean", ""),
            "test_weighted_f1_std": full_sum.get("test_weighted_f1_std", ""),
            "test_confusion_matrix_sum": json.dumps(full_cm),
        }
    )

    if do_per_partial:
        for pidx in range(1, int(num_splits) + 1):
            LOGGER.info("step=build_partial pidx=%d/%d", int(pidx), int(num_splits))
            rc_b = build_partial_amotf_images_from_csv(
                labels_csv=str(labels_csv),
                out_size=int(out_size),
                m=int(m),
                tau=int(tau),
                spans=list(spans),
                max_points=int(max_points),
                overwrite=bool(overwrite),
                num_workers=int(num_workers_build),
                num_splits=int(num_splits),
                partial_index=int(pidx),
                partial_phase=str(partial_phase),
            )
            if rc_b != 0:
                LOGGER.error("partial_build_failed pidx=%d rc=%d", int(pidx), int(rc_b))
                continue

            labels_p = root / f"labels_partial_p{int(pidx):02d}.csv"
            stats = _write_partial_labels_csv(
                str(labels_csv),
                out_csv=labels_p,
                num_splits=int(num_splits),
                partial_index=int(pidx),
                partial_phase=str(partial_phase),
                max_points=int(max_points),
                input_mode=str(train_input_mode),
            )
            if int(stats.get("keep", 0)) < 20:
                LOGGER.error("Too few usable partial samples pidx=%d stats=%s", int(pidx), stats)
                continue

            LOGGER.info("step=train_partial pidx=%d/%d", int(pidx), int(num_splits))
            rc_t = train_dino_soh_classifier(
                labels_csv=str(labels_p),
                runs_root=str(runs_root),
                run_name=f"{base}/partial_p{int(pidx):02d}",
                input_mode=str(train_input_mode),
                max_points=int(max_points),
                kfold=int(kfold),
                split_indices_json="",
                exclude_samples_txt=str(exclude_samples_txt),
                seed=int(seed),
                img_size=int(img_size),
                batch_size=int(batch_size),
                epochs=int(epochs),
                lr=float(lr),
                weight_decay=float(weight_decay),
                lr_scheduler=str(lr_scheduler),
                lr_warmup_ratio=float(lr_warmup_ratio),
                lr_min=float(lr_min),
                lr_plateau_factor=float(lr_plateau_factor),
                lr_plateau_patience=int(lr_plateau_patience),
                backbone_lr_mult=float(backbone_lr_mult),
                num_workers=int(num_workers),
                fusion=str(fusion),
                val_ratio=float(val_ratio),
                metric_for_best=str(metric_for_best),
                confidence_gap_threshold=float(confidence_gap_threshold),
                finetune_backbone=bool(finetune_backbone),
                chan_proj=str(chan_proj),
                chan_hidden=int(chan_hidden),
                chan_norm=str(chan_norm),
                chan_dropout=float(chan_dropout),
                use_lora=bool(use_lora),
                npz_norm=str(npz_norm),
                npz_global_max_log=float(npz_global_max_log),
                use_class_weights=bool(use_class_weights),
                lora_backend=str(lora_backend),
                lora_r=int(lora_r),
                lora_alpha=float(lora_alpha),
                lora_dropout=float(lora_dropout),
                lora_targets=str(lora_targets),
                allow_fallback_backbone=bool(allow_fallback_backbone),
                hf_model_id=str(hf_model_id),
                hf_local_only=bool(hf_local_only),
            )
            if rc_t != 0:
                LOGGER.error("partial_train_failed pidx=%d rc=%d", int(pidx), int(rc_t))
                continue

            rdir = Path(runs_root) / base / f"partial_p{int(pidx):02d}" / str(train_input_mode)
            summ: Dict[str, Any] = {}
            if (rdir / "summary.json").exists():
                with open(rdir / "summary.json", "r", encoding="utf-8") as f:
                    summ = json.load(f)
            cm = _sum_confusion_from_fold_metrics(rdir / "fold_metrics.csv")
            sweep_rows.append(
                {
                    "partial_index": int(pidx),
                    "partial_ratio": float(pidx) / float(num_splits),
                    "variant": "partial",
                    "n_samples": int(stats.get("keep", 0)),
                    "test_acc_mean": summ.get("test_acc_mean", ""),
                    "test_acc_std": summ.get("test_acc_std", ""),
                    "test_macro_f1_mean": summ.get("test_macro_f1_mean", ""),
                    "test_macro_f1_std": summ.get("test_macro_f1_std", ""),
                    "test_weighted_f1_mean": summ.get("test_weighted_f1_mean", ""),
                    "test_weighted_f1_std": summ.get("test_weighted_f1_std", ""),
                    "test_confusion_matrix_sum": json.dumps(cm),
                }
            )

    if do_pooled and (not do_per_partial):
        thrs = [int(x) for x in list(pooled_thresholds or []) if int(x) > 0]
        if thrs:
            pmin_build = max(1, min(thrs))
            for pidx in range(int(pmin_build), int(num_splits) + 1):
                LOGGER.info("step=build_partial_for_pooled pidx=%d/%d", int(pidx), int(num_splits))
                rc_b = build_partial_amotf_images_from_csv(
                    labels_csv=str(labels_csv),
                    out_size=int(out_size),
                    m=int(m),
                    tau=int(tau),
                    spans=list(spans),
                    max_points=int(max_points),
                    overwrite=bool(overwrite),
                    num_workers=int(num_workers_build),
                    num_splits=int(num_splits),
                    partial_index=int(pidx),
                    partial_phase=str(partial_phase),
                )
                if rc_b != 0:
                    LOGGER.error("partial_build_failed pidx=%d rc=%d", int(pidx), int(rc_b))
                    continue

    _write_csv(sweep_rows, root / "partial_sweep_table.csv")
    with open(root / "partial_sweep_table.json", "w", encoding="utf-8") as f:
        json.dump(sweep_rows, f, ensure_ascii=False, indent=2)

    full_acc = _safe_float(full_sum.get("test_acc_mean", float("nan")))
    full_macro_f1 = _safe_float(full_sum.get("test_macro_f1_mean", float("nan")))
    full_weighted_f1 = _safe_float(full_sum.get("test_weighted_f1_mean", float("nan")))

    partial_only = [r for r in sweep_rows if str(r.get("variant", "")) == "partial"]
    partial_only = sorted(partial_only, key=lambda r: float(r.get("partial_ratio", 0.0)))
    analysis_rows: List[Dict[str, Any]] = []
    for r in partial_only:
        pr = float(r.get("partial_ratio", 0.0))
        a = _safe_float(r.get("test_acc_mean", float("nan")))
        mf = _safe_float(r.get("test_macro_f1_mean", float("nan")))
        wf = _safe_float(r.get("test_weighted_f1_mean", float("nan")))
        analysis_rows.append(
            {
                "partial_index": int(r.get("partial_index", 0)),
                "partial_ratio": pr,
                "n_samples": r.get("n_samples", ""),
                "test_acc_mean": r.get("test_acc_mean", ""),
                "test_macro_f1_mean": r.get("test_macro_f1_mean", ""),
                "test_weighted_f1_mean": r.get("test_weighted_f1_mean", ""),
                "delta_acc_vs_full": float(full_acc - a) if np.isfinite(full_acc) and np.isfinite(a) else "",
                "delta_macro_f1_vs_full": float(full_macro_f1 - mf) if np.isfinite(full_macro_f1) and np.isfinite(mf) else "",
                "delta_weighted_f1_vs_full": float(full_weighted_f1 - wf) if np.isfinite(full_weighted_f1) and np.isfinite(wf) else "",
            }
        )

    _write_csv(analysis_rows, root / "partial_sweep_analysis.csv")
    with open(root / "partial_sweep_analysis.json", "w", encoding="utf-8") as f:
        json.dump(analysis_rows, f, ensure_ascii=False, indent=2)

    pooled_rows: List[Dict[str, Any]] = []
    if do_pooled:
        for thr in pooled_thresholds:
            LOGGER.info("step=train_partial_ge thr=%d", int(thr))
            labels_ge = root / f"labels_partial_ge_p{int(thr):02d}.csv"
            stats_ge = _write_partial_labels_ge_csv(
                str(labels_csv),
                out_csv=labels_ge,
                num_splits=int(num_splits),
                partial_index_min=int(thr),
                partial_phase=str(partial_phase),
                max_points=int(max_points),
                input_mode=str(train_input_mode),
            )
            if int(stats_ge.get("keep", 0)) < 20:
                LOGGER.error("Too few usable partial>=thr samples thr=%d stats=%s", int(thr), stats_ge)
                continue

            rc_ge = train_dino_soh_classifier(
                labels_csv=str(labels_ge),
                runs_root=str(runs_root),
                run_name=f"{base}/partial_ge_p{int(thr):02d}",
                input_mode=str(train_input_mode),
                max_points=int(max_points),
                kfold=int(kfold),
                split_indices_json="",
                exclude_samples_txt=str(exclude_samples_txt),
                seed=int(seed),
                img_size=int(img_size),
                batch_size=int(batch_size),
                epochs=int(epochs),
                lr=float(lr),
                weight_decay=float(weight_decay),
                lr_scheduler=str(lr_scheduler),
                lr_warmup_ratio=float(lr_warmup_ratio),
                lr_min=float(lr_min),
                lr_plateau_factor=float(lr_plateau_factor),
                lr_plateau_patience=int(lr_plateau_patience),
                backbone_lr_mult=float(backbone_lr_mult),
                num_workers=int(num_workers),
                fusion=str(fusion),
                val_ratio=float(val_ratio),
                metric_for_best=str(metric_for_best),
                confidence_gap_threshold=float(confidence_gap_threshold),
                finetune_backbone=bool(finetune_backbone),
                chan_proj=str(chan_proj),
                chan_hidden=int(chan_hidden),
                chan_norm=str(chan_norm),
                chan_dropout=float(chan_dropout),
                use_lora=bool(use_lora),
                lora_backend=str(lora_backend),
                lora_r=int(lora_r),
                lora_alpha=float(lora_alpha),
                lora_dropout=float(lora_dropout),
                lora_targets=str(lora_targets),
                allow_fallback_backbone=bool(allow_fallback_backbone),
                hf_model_id=str(hf_model_id),
                hf_local_only=bool(hf_local_only),
                group_by_sample_id=True,
            )
            if rc_ge != 0:
                LOGGER.error("partial_ge_train_failed thr=%d rc=%d", int(thr), int(rc_ge))
                continue

            rdir = Path(runs_root) / base / f"partial_ge_p{int(thr):02d}" / str(train_input_mode)
            summ: Dict[str, Any] = {}
            if (rdir / "summary.json").exists():
                with open(rdir / "summary.json", "r", encoding="utf-8") as f:
                    summ = json.load(f)
            cm = _sum_confusion_from_fold_metrics(rdir / "fold_metrics.csv")
            pooled_rows.append(
                {
                    "partial_index_min": int(thr),
                    "variant": "partial_ge",
                    "n_samples": int(stats_ge.get("keep", 0)),
                    "test_acc_mean": summ.get("test_acc_mean", ""),
                    "test_acc_std": summ.get("test_acc_std", ""),
                    "test_macro_f1_mean": summ.get("test_macro_f1_mean", ""),
                    "test_macro_f1_std": summ.get("test_macro_f1_std", ""),
                    "test_weighted_f1_mean": summ.get("test_weighted_f1_mean", ""),
                    "test_weighted_f1_std": summ.get("test_weighted_f1_std", ""),
                    "test_confusion_matrix_sum": json.dumps(cm),
                }
            )

    if pooled_rows:
        _write_csv(pooled_rows, root / "partial_threshold_table.csv")
        with open(root / "partial_threshold_table.json", "w", encoding="utf-8") as f:
            json.dump(pooled_rows, f, ensure_ascii=False, indent=2)

    return 0


def _write_labels_subset(in_labels_csv: str, original_paths_keep: set, out_csv: Path) -> None:
    rows = load_label_rows(in_labels_csv)
    keep_rows: List[Dict[str, Any]] = []
    for r in rows:
        op = remap_known_root(str(r.get("original_path", "")).strip())
        if op in original_paths_keep:
            keep_rows.append(
                {
                    "sample_id": str(r.get("sample_id", "")).strip(),
                    "original_path": op,
                    "assigned_class": str(r.get("assigned_class", "")).strip(),
                }
            )
    _write_csv(keep_rows, out_csv)


def _collect_intersection_original_paths(labels_csv: str, *, max_points: int = 2000) -> Tuple[set, Dict[str, int]]:
    rows = load_label_rows(labels_csv)
    keep: set = set()
    stats: Dict[str, int] = {
        "total": 0,
        "keep": 0,
        "missing_png": 0,
        "missing_amotf": 0,
        "missing_both": 0,
    }
    for r in rows:
        op = remap_known_root(str(r.get("original_path", "")).strip())
        if not op:
            continue
        stats["total"] += 1
        png_ok = _exists(os.path.join(op, "charge.png")) and _exists(os.path.join(op, "discharge.png"))
        c_paths = _amotf_output_paths(op, max_points=int(max_points))
        cma_ok = _is_valid_npz(str(c_paths["c_npz"])) and _is_valid_npz(str(c_paths["d_npz"]))
        if png_ok and cma_ok:
            keep.add(op)
            stats["keep"] += 1
        elif (not png_ok) and (not cma_ok):
            stats["missing_both"] += 1
        elif not png_ok:
            stats["missing_png"] += 1
        else:
            stats["missing_amotf"] += 1
    return keep, stats


def _compare_runs(run_root: Path) -> None:
    cma_sum = run_root / "amotf" / "summary.json"
    png_sum = run_root / "png" / "summary.json"

    def _load_json(p: Path) -> Dict[str, Any]:
        if not p.exists():
            return {}
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    c = _load_json(cma_sum)
    p = _load_json(png_sum)
    out = {
        "amotf": c,
        "png": p,
    }
    with open(run_root / "compare_summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    rows: List[Dict[str, Any]] = []
    for method, d in [("amotf", c), ("png", p)]:
        if not d:
            continue
        rows.append(
            {
                "method": method,
                "kfold": d.get("kfold", ""),
                "test_acc_mean": d.get("test_acc_mean", ""),
                "test_acc_std": d.get("test_acc_std", ""),
                "test_macro_f1_mean": d.get("test_macro_f1_mean", ""),
                "test_macro_f1_std": d.get("test_macro_f1_std", ""),
                "test_weighted_f1_mean": d.get("test_weighted_f1_mean", ""),
                "test_weighted_f1_std": d.get("test_weighted_f1_std", ""),
            }
        )
    _write_csv(rows, run_root / "compare_table.csv")


# ----------------------------
# CLI
# ----------------------------


def _parse_spans(s: str) -> List[int]:
    out: List[int] = []
    for x in str(s).split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="AMOTF image build + DINOv3(frozen backbone) SOH classification + PNG baseline compare (joint stratified by dataset×class)."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_build = sub.add_parser("build_amotf", help="Build AMOTF images (PNG/NPZ) under each sample dir/amotf/")
    ap_build.add_argument("--labels_csv", type=str, default=_default_labels_csv())
    ap_build.add_argument("--out_size", type=int, default=0)
    ap_build.add_argument("--m", type=int, default=5)
    ap_build.add_argument("--tau", type=int, default=1)
    ap_build.add_argument("--spans", type=str, default="1,2,4,8")
    ap_build.add_argument("--max_points", type=int, default=2000)
    ap_build.add_argument("--overwrite", action="store_true")
    ap_build.add_argument("--num_workers", type=int, default=0)

    ap_train = sub.add_parser("train", help="Train classifier (use --split_indices_json to run a single fixed split)")
    ap_train.add_argument("--labels_csv", type=str, default=_default_labels_csv())
    ap_train.add_argument("--runs_root", type=str, default=_default_runs_root())
    ap_train.add_argument("--run_name", type=str, default="")
    ap_train.add_argument("--input_mode", choices=["amotf", "amotf_npz", "png"], default="amotf_npz")
    ap_train.add_argument("--max_points", type=int, default=2000)
    ap_train.add_argument("--kfold", type=int, default=5)
    ap_train.add_argument("--split_indices_json", type=str, default="")
    ap_train.add_argument("--exclude_samples_txt", type=str, default=_default_exclude_samples_txt())
    ap_train.add_argument("--seed", type=int, default=42)
    ap_train.add_argument("--img_size", type=int, default=224)
    ap_train.add_argument("--batch_size", type=int, default=24)
    ap_train.add_argument("--epochs", type=int, default=10)
    ap_train.add_argument("--lr", type=float, default=5e-4)
    ap_train.add_argument("--weight_decay", type=float, default=1e-4)
    ap_train.add_argument("--lr_scheduler", choices=["none", "cosine", "cosine_warmup", "onecycle", "plateau"], default="none")
    ap_train.add_argument("--lr_warmup_ratio", type=float, default=0.0)
    ap_train.add_argument("--lr_min", type=float, default=0.0)
    ap_train.add_argument("--lr_plateau_factor", type=float, default=0.5)
    ap_train.add_argument("--lr_plateau_patience", type=int, default=2)
    ap_train.add_argument("--backbone_lr_mult", type=float, default=1.0)
    ap_train.add_argument("--num_workers", type=int, default=4)
    ap_train.add_argument("--fusion", choices=["concat", "stack"], default="concat")
    ap_train.add_argument("--val_ratio", type=float, default=0.2)
    ap_train.add_argument("--metric_for_best", choices=["macro_f1", "acc"], default="macro_f1")
    ap_train.add_argument("--confidence_gap_threshold", type=float, default=0.3)
    ap_train.add_argument("--chan_proj", choices=["mlp", "linear"], default="mlp")
    ap_train.add_argument("--chan_hidden", type=int, default=0)
    ap_train.add_argument("--chan_norm", choices=["group", "none"], default="group")
    ap_train.add_argument("--chan_dropout", type=float, default=0.0)
    ap_train.add_argument("--npz_norm", choices=["log1p_global", "per_sample_minmax", "none"], default="log1p_global")
    ap_train.add_argument("--npz_global_max_log", type=float, default=10.0)
    ap_train.add_argument("--finetune_backbone", action="store_true")
    ap_train.add_argument("--use_class_weights", action="store_true")
    ap_train.add_argument("--use_lora", action="store_true")
    ap_train.add_argument("--lora_backend", choices=["auto", "peft", "custom"], default="auto")
    ap_train.add_argument("--lora_r", type=int, default=8)
    ap_train.add_argument("--lora_alpha", type=float, default=16.0)
    ap_train.add_argument("--lora_dropout", type=float, default=0.0)
    ap_train.add_argument("--lora_targets", type=str, default="pwconv,fc,qkv,proj")
    ap_train.add_argument("--allow_fallback_backbone", action="store_true")
    ap_train.add_argument("--hf_model_id", type=str, default="facebook/dinov3-convnext-tiny-pretrain-lvd1689m")
    ap_train.add_argument("--hf_local_only", action="store_true")

    ap_all = sub.add_parser("run_all", help="Build AMOTF then train both AMOTF and PNG baseline under one run dir (supports --split_indices_json and --exclude_samples_txt)")
    ap_all.add_argument("--labels_csv", type=str, default=_default_labels_csv())
    ap_all.add_argument("--runs_root", type=str, default=_default_runs_root())
    ap_all.add_argument("--run_name", type=str, default="")
    ap_all.add_argument("--out_size", type=int, default=0)
    ap_all.add_argument("--m", type=int, default=5)
    ap_all.add_argument("--tau", type=int, default=1)
    ap_all.add_argument("--spans", type=str, default="1,2,4,8")
    ap_all.add_argument("--max_points", type=int, default=2000)
    ap_all.add_argument("--overwrite", action="store_true")
    ap_all.add_argument("--num_workers_build", type=int, default=0)
    ap_all.add_argument("--kfold", type=int, default=5)
    ap_all.add_argument("--split_indices_json", type=str, default="")
    ap_all.add_argument("--exclude_samples_txt", type=str, default=_default_exclude_samples_txt())
    ap_all.add_argument("--seed", type=int, default=42)
    ap_all.add_argument("--img_size", type=int, default=224)
    ap_all.add_argument("--batch_size", type=int, default=24)
    ap_all.add_argument("--epochs", type=int, default=10)
    ap_all.add_argument("--lr", type=float, default=5e-4)
    ap_all.add_argument("--weight_decay", type=float, default=1e-4)
    ap_all.add_argument("--lr_scheduler", choices=["none", "cosine", "cosine_warmup", "onecycle", "plateau"], default="none")
    ap_all.add_argument("--lr_warmup_ratio", type=float, default=0.0)
    ap_all.add_argument("--lr_min", type=float, default=0.0)
    ap_all.add_argument("--lr_plateau_factor", type=float, default=0.5)
    ap_all.add_argument("--lr_plateau_patience", type=int, default=2)
    ap_all.add_argument("--backbone_lr_mult", type=float, default=1.0)
    ap_all.add_argument("--num_workers", type=int, default=4)
    ap_all.add_argument("--fusion", choices=["concat", "stack"], default="concat")
    ap_all.add_argument("--val_ratio", type=float, default=0.2)
    ap_all.add_argument("--metric_for_best", choices=["macro_f1", "acc"], default="macro_f1")
    ap_all.add_argument("--confidence_gap_threshold", type=float, default=0.3)
    ap_all.add_argument("--chan_proj", choices=["mlp", "linear"], default="mlp")
    ap_all.add_argument("--chan_hidden", type=int, default=0)
    ap_all.add_argument("--chan_norm", choices=["group", "none"], default="group")
    ap_all.add_argument("--chan_dropout", type=float, default=0.0)
    ap_all.add_argument("--npz_norm", choices=["log1p_global", "per_sample_minmax", "none"], default="log1p_global")
    ap_all.add_argument("--npz_global_max_log", type=float, default=8.0)
    ap_all.add_argument("--finetune_backbone", action="store_true")
    ap_all.add_argument("--use_class_weights", action="store_true")
    ap_all.add_argument("--use_lora", action="store_true")
    ap_all.add_argument("--lora_backend", choices=["auto", "peft", "custom"], default="auto")
    ap_all.add_argument("--lora_r", type=int, default=8)
    ap_all.add_argument("--lora_alpha", type=float, default=16.0)
    ap_all.add_argument("--lora_dropout", type=float, default=0.0)
    ap_all.add_argument("--lora_targets", type=str, default="pwconv,fc,qkv,proj")
    ap_all.add_argument("--allow_fallback_backbone", action="store_true")
    ap_all.add_argument("--hf_model_id", type=str, default="facebook/dinov3-convnext-tiny-pretrain-lvd1689m")
    ap_all.add_argument("--hf_local_only", action="store_true")

    args = ap.parse_args()
    # Normalize cross-OS paths in CLI args
    if hasattr(args, "labels_csv"):
        args.labels_csv = remap_known_root(args.labels_csv)
    if hasattr(args, "runs_root"):
        args.runs_root = remap_known_root(args.runs_root)
    if hasattr(args, "split_indices_json"):
        args.split_indices_json = remap_known_root(getattr(args, "split_indices_json", ""))
    if hasattr(args, "exclude_samples_txt"):
        args.exclude_samples_txt = remap_known_root(getattr(args, "exclude_samples_txt", ""))

    # Dispatch will be filled once implementations are patched in.
    if args.cmd == "build_amotf":
        spans = _parse_spans(args.spans)
        return build_amotf_images_from_csv(
            labels_csv=args.labels_csv,
            out_size=args.out_size,
            m=args.m,
            tau=args.tau,
            spans=spans,
            max_points=args.max_points,
            overwrite=bool(args.overwrite),
            num_workers=int(args.num_workers),
        )

    if args.cmd == "train":
        _ensure_sklearn()
        return train_dino_soh_classifier(**vars(args))

    if args.cmd == "run_all":
        _ensure_sklearn()
        spans = _parse_spans(args.spans)
        run_name = str(args.run_name).strip() or _now_run_name()
        run_root = Path(args.runs_root) / run_name
        _ensure_dir(run_root)
        _setup_logging(run_root / "run_all.log")

        LOGGER.info("run_all run_name=%s", run_name)
        LOGGER.info("step=build_amotf")
        rc = build_amotf_images_from_csv(
            labels_csv=args.labels_csv,
            out_size=int(args.out_size),
            m=int(args.m),
            tau=int(args.tau),
            spans=spans,
            max_points=int(args.max_points),
            overwrite=bool(args.overwrite),
            num_workers=int(args.num_workers_build),
        )
        if rc != 0:
            LOGGER.error("build_amotf failed rc=%d", rc)
            return int(rc)

        keep, stats = _collect_intersection_original_paths(args.labels_csv, max_points=int(args.max_points))
        LOGGER.info("intersection_stats=%s", stats)
        if int(stats.get("keep", 0)) < 20:
            LOGGER.error("Too few intersection samples: %s", stats)
            return 2

        subset_csv = run_root / "labels_intersection.csv"
        _write_labels_subset(args.labels_csv, keep, subset_csv)

        LOGGER.info("step=train_amotf_npz")
        rc1 = train_dino_soh_classifier(
            labels_csv=str(subset_csv),
            runs_root=str(args.runs_root),
            run_name=run_name,
            input_mode="amotf_npz",
            max_points=int(args.max_points),
            kfold=int(args.kfold),
            split_indices_json=str(getattr(args, "split_indices_json", "")),
            exclude_samples_txt=str(getattr(args, "exclude_samples_txt", "")),
            seed=int(args.seed),
            img_size=int(args.img_size),
            batch_size=int(args.batch_size),
            epochs=int(args.epochs),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            lr_scheduler=str(getattr(args, "lr_scheduler", "none")),
            lr_warmup_ratio=float(getattr(args, "lr_warmup_ratio", 0.0)),
            lr_min=float(getattr(args, "lr_min", 0.0)),
            lr_plateau_factor=float(getattr(args, "lr_plateau_factor", 0.5)),
            lr_plateau_patience=int(getattr(args, "lr_plateau_patience", 2)),
            backbone_lr_mult=float(getattr(args, "backbone_lr_mult", 1.0)),
            num_workers=int(args.num_workers),
            fusion=str(args.fusion),
            val_ratio=float(args.val_ratio),
            metric_for_best=str(args.metric_for_best),
            confidence_gap_threshold=float(args.confidence_gap_threshold),
            finetune_backbone=bool(args.finetune_backbone),
            chan_proj=str(getattr(args, "chan_proj", "mlp")),
            chan_hidden=int(getattr(args, "chan_hidden", 0)),
            chan_norm=str(getattr(args, "chan_norm", "group")),
            chan_dropout=float(getattr(args, "chan_dropout", 0.0)),
            npz_norm=str(getattr(args, "npz_norm", "log1p_global")),
            npz_global_max_log=float(getattr(args, "npz_global_max_log", 10.0)),
            use_lora=bool(getattr(args, "use_lora", False)),
            use_class_weights=bool(getattr(args, "use_class_weights", False)),
            lora_backend=str(getattr(args, "lora_backend", "auto")),
            lora_r=int(getattr(args, "lora_r", 8)),
            lora_alpha=float(getattr(args, "lora_alpha", 16.0)),
            lora_dropout=float(getattr(args, "lora_dropout", 0.0)),
            lora_targets=str(getattr(args, "lora_targets", "pwconv,fc,qkv,proj")),
            allow_fallback_backbone=bool(args.allow_fallback_backbone),
            hf_model_id=str(args.hf_model_id),
            hf_local_only=bool(args.hf_local_only),
        )
        if rc1 != 0:
            return int(rc1)

        LOGGER.info("step=train_png")
        rc2 = train_dino_soh_classifier(
            labels_csv=str(subset_csv),
            runs_root=str(args.runs_root),
            run_name=run_name,
            input_mode="png",
            max_points=int(args.max_points),
            kfold=int(args.kfold),
            split_indices_json=str(getattr(args, "split_indices_json", "")),
            exclude_samples_txt=str(getattr(args, "exclude_samples_txt", "")),
            seed=int(args.seed),
            img_size=int(args.img_size),
            batch_size=int(args.batch_size),
            epochs=int(args.epochs),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            lr_scheduler=str(getattr(args, "lr_scheduler", "none")),
            lr_warmup_ratio=float(getattr(args, "lr_warmup_ratio", 0.0)),
            lr_min=float(getattr(args, "lr_min", 0.0)),
            lr_plateau_factor=float(getattr(args, "lr_plateau_factor", 0.5)),
            lr_plateau_patience=int(getattr(args, "lr_plateau_patience", 2)),
            backbone_lr_mult=float(getattr(args, "backbone_lr_mult", 1.0)),
            num_workers=int(args.num_workers),
            fusion=str(args.fusion),
            val_ratio=float(args.val_ratio),
            metric_for_best=str(args.metric_for_best),
            confidence_gap_threshold=float(args.confidence_gap_threshold),
            finetune_backbone=bool(args.finetune_backbone),
            chan_proj=str(getattr(args, "chan_proj", "mlp")),
            chan_hidden=int(getattr(args, "chan_hidden", 0)),
            chan_norm=str(getattr(args, "chan_norm", "group")),
            chan_dropout=float(getattr(args, "chan_dropout", 0.0)),
            npz_norm=str(getattr(args, "npz_norm", "log1p_global")),
            npz_global_max_log=float(getattr(args, "npz_global_max_log", 10.0)),
            use_lora=bool(getattr(args, "use_lora", False)),
            use_class_weights=bool(getattr(args, "use_class_weights", False)),
            lora_backend=str(getattr(args, "lora_backend", "auto")),
            lora_r=int(getattr(args, "lora_r", 8)),
            lora_alpha=float(getattr(args, "lora_alpha", 16.0)),
            lora_dropout=float(getattr(args, "lora_dropout", 0.0)),
            lora_targets=str(getattr(args, "lora_targets", "pwconv,fc,qkv,proj")),
            allow_fallback_backbone=bool(args.allow_fallback_backbone),
            hf_model_id=str(args.hf_model_id),
            hf_local_only=bool(args.hf_local_only),
        )
        if rc2 != 0:
            return int(rc2)

        _compare_runs(run_root)
        LOGGER.info("run_all finished. compare_table=%s", str(run_root / "compare_table.csv"))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
