#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import hashlib
import json
import logging
import os
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


LOGGER = logging.getLogger("soh_cmaotn_dino_dataset_sweep")


TARGET_DATASETS = ["CALCE", "HNEI", "Oxford", "SNL_LFP", "SNL_NCA", "SNL_NMC", "UL_Purdue"]


SWEEP_STEPS: List[Dict[str, Any]] = [
    {
        "step": 1,
        "new_dataset": "HNEI",
        "rationale": "Benchmark: only large dataset (10k+) covering A/B/C; establishes baseline generalization.",
        "expected_dist": "75% : 17% : 8%",
    },
    {
        "step": 2,
        "new_dataset": "SNL_LFP",
        "rationale": "Balancing + chemistry shift: HNEI (A-heavy) complemented by SNL_LFP (0% A; mostly B/C).",
        "expected_dist": "58% : 21% : 21%",
    },
    {
        "step": 3,
        "new_dataset": "SNL_NMC",
        "rationale": "Reinforcing transition: SNL_NMC is B-dominant, strengthening mid-SOH discrimination.",
        "expected_dist": "53% : 27% : 20%",
    },
    {
        "step": 4,
        "new_dataset": "SNL_NCA",
        "rationale": "Material diversity: add NCA system with balanced A/B to smooth NMC vs NCA domain shift.",
        "expected_dist": "53% : 27% : 19%",
    },
    {
        "step": 5,
        "new_dataset": "Oxford",
        "rationale": "Corner case (EOL): small but C-heavy dataset to improve end-of-life recall.",
        "expected_dist": "52% : 28% : 20%",
    },
    {
        "step": 6,
        "new_dataset": "CALCE",
        "rationale": "Noise resistance: CALCE is ~all-A; stress-test whether model collapses to conservative A prediction.",
        "expected_dist": "64% : 21% : 15%",
    },
    {
        "step": 7,
        "new_dataset": "UL_Purdue",
        "rationale": "Full scale: final small all-A dataset; complete all-datasets setting for final validation.",
        "expected_dist": "65% : 21% : 14%",
    },
]


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


def _sha1_short(s: str, n: int = 10) -> str:
    return hashlib.sha1(str(s).encode("utf-8", errors="ignore")).hexdigest()[: int(n)]


def _write_csv_rows(rows: List[Dict[str, Any]], path: Path) -> None:
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


def _append_result_row(path: Path, row: Dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    exists = path.exists() and (path.stat().st_size > 0)

    if exists:
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
            rdr = csv.DictReader(f)
            header = list(rdr.fieldnames or [])
    else:
        header = []

    for k in row.keys():
        if k not in header:
            header.append(k)

    if (not exists) or (set(header) != set((row.keys()))):
        # Rewrite file with union header (avoid breaking when new cols are added)
        old_rows: List[Dict[str, Any]] = []
        if exists:
            with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
                old_rows = [dict(r) for r in csv.DictReader(f)]
        old_rows.append(row)
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for r in old_rows:
                w.writerow(r)
        return

    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow(row)


def _import_reference_module():
    # Ensure current script dir is importable
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    import soh_dino_cmaotn_npz_partial_cycles_partial_sweep_soc_horizontal as ref

    return ref


def run_dataset_sweep(
    *,
    labels_csv: str,
    runs_root: str,
    run_name: str,
    max_points: int,
    kfold: int,
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
    finetune_backbone: bool = True,
    lr_scheduler: str = "cosine_warmup",
    lr_warmup_ratio: float = 0.1,
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
    allow_fallback_backbone: bool = False,
    hf_model_id: str = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    hf_local_only: bool = False,
    use_class_weights: bool = False,
    min_samples: int = 21,
    **_: Any,
) -> int:
    ref = _import_reference_module()
    ref._ensure_sklearn()

    labels_csv = ref.remap_known_root(labels_csv)
    runs_root = ref.remap_known_root(runs_root)

    if not run_name:
        run_name = ref._now_run_name()

    top_run_root = Path(runs_root) / str(run_name)
    _ensure_dir(top_run_root)
    _setup_logging(top_run_root / "dataset_sweep.log")

    LOGGER.info("dataset_sweep run_root=%s", str(top_run_root))
    LOGGER.info("labels_csv=%s", str(labels_csv))
    LOGGER.info("target_datasets=%s", TARGET_DATASETS)

    # Load once, and require usable cmaotn_npz.
    records_all, dropped0 = ref._load_records(labels_csv, max_points=int(max_points))
    records, dropped1 = ref._filter_by_input_mode(records_all, input_mode="cmaotn_npz")

    dropped = dict(dropped0)
    for k, v in dropped1.items():
        dropped[k] = dropped.get(k, 0) + int(v)

    LOGGER.info("loaded_records=%d usable_records=%d dropped=%s", len(records_all), len(records), dropped)

    results_csv = top_run_root / "dataset_sweep_results.csv"

    included: List[str] = []
    for st in SWEEP_STEPS:
        step = int(st.get("step", 0))
        new_ds = str(st.get("new_dataset", "")).strip()
        if not new_ds:
            continue
        included.append(new_ds)

        comb = tuple(included)
        comb_name = "|".join(comb)
        comb_hash = _sha1_short(comb_name, n=10)
        combo_dir = top_run_root / f"s{int(step):02d}_{new_ds}_{comb_hash}"

        sel = set(comb)
        subset_records = [r for r in records if str(getattr(r, "dataset_name", "")) in sel]
        sample_count = int(len(subset_records))
        if sample_count < int(min_samples):
            LOGGER.info(
                "skip step=%d comb=%s samples=%d (<min_samples=%d)",
                int(step),
                comb_name,
                int(sample_count),
                int(min_samples),
            )
            continue

        # Pre-check joint stratification feasibility: each dataset|class stratum must have >= kfold.
        joint_keys = ref._make_joint_keys(subset_records)
        uniq, cnt = np.unique(joint_keys, return_counts=True)
        min_cnt = int(cnt.min()) if cnt.size else 0
        if int(kfold) > 1 and min_cnt < int(kfold):
            LOGGER.info(
                "skip step=%d comb=%s: some dataset×class strata too small for kfold (min_count=%d kfold=%d)",
                int(step),
                comb_name,
                int(min_cnt),
                int(kfold),
            )
            continue

        class_counts: Dict[str, int] = {"A": 0, "B": 0, "C": 0}
        for r in subset_records:
            cc = str(getattr(r, "assigned_class", ""))
            if cc in class_counts:
                class_counts[cc] += 1

        tot_abc = int(sum(class_counts.values()))
        a = int(class_counts.get("A", 0))
        b = int(class_counts.get("B", 0))
        c = int(class_counts.get("C", 0))
        a_pct = (float(a) / float(tot_abc) * 100.0) if tot_abc else 0.0
        b_pct = (float(b) / float(tot_abc) * 100.0) if tot_abc else 0.0
        c_pct = (float(c) / float(tot_abc) * 100.0) if tot_abc else 0.0
        actual_dist = f"{a_pct:.2f}% : {b_pct:.2f}% : {c_pct:.2f}%"

        _ensure_dir(combo_dir)

        subset_csv = combo_dir / "temp_current_sweep.csv"
        _write_csv_rows(
            [
                {
                    "sample_id": str(r.sample_id),
                    "original_path": str(r.original_path),
                    "assigned_class": str(r.assigned_class),
                }
                for r in subset_records
            ],
            subset_csv,
        )

        LOGGER.info(
            "run step=%d comb=%s samples=%d A/B/C=%d/%d/%d actual=%s dir=%s",
            int(step),
            comb_name,
            int(sample_count),
            int(a),
            int(b),
            int(c),
            actual_dist,
            str(combo_dir),
        )

        rc = ref.train_dino_soh_classifier(
            labels_csv=str(subset_csv),
            runs_root=str(top_run_root),
            run_name=str(combo_dir.name),
            input_mode="cmaotn_npz",
            max_points=int(max_points),
            kfold=int(kfold),
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

        summ_path = combo_dir / "cmaotn_npz" / "summary.json"
        summ: Dict[str, Any] = {}
        if summ_path.exists():
            try:
                with open(summ_path, "r", encoding="utf-8") as f:
                    summ = json.load(f)
            except Exception:
                summ = {}

        row = {
            "step": int(step),
            "new_dataset": str(new_ds),
            "num_datasets": int(len(comb)),
            "datasets": "+".join(comb),
            "rationale": str(st.get("rationale", "")),
            "expected_dist": str(st.get("expected_dist", "")),
            "actual_dist": str(actual_dist),
            "A": int(a),
            "B": int(b),
            "C": int(c),
            "A_pct": float(a_pct),
            "B_pct": float(b_pct),
            "C_pct": float(c_pct),
            "sample_count": int(sample_count),
            "run_dir": str(combo_dir),
            "train_rc": int(rc),
            "test_acc_mean": float(summ.get("test_acc_mean", float("nan"))),
            "test_acc_std": float(summ.get("test_acc_std", float("nan"))),
            "test_macro_f1_mean": float(summ.get("test_macro_f1_mean", float("nan"))),
        }
        _append_result_row(results_csv, row)

    LOGGER.info("done. results_csv=%s", str(results_csv))
    return 0


def main() -> int:
    ref = _import_reference_module()

    ap = argparse.ArgumentParser(
        description="CMAOTN NPZ build + dataset combination sweep for DINO SOH classification (joint stratified by dataset×class)."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_build = sub.add_parser("build_cmaotn", help="Build CMAOTN NPZ (and optional PNG) under each sample dir/cmaotn/")
    ap_build.add_argument("--labels_csv", type=str, default=ref._default_labels_csv())
    ap_build.add_argument("--out_size", type=int, default=0)
    ap_build.add_argument("--m", type=int, default=5)
    ap_build.add_argument("--tau", type=int, default=1)
    ap_build.add_argument("--spans", type=str, default="1,2,4,8")
    ap_build.add_argument("--max_points", type=int, default=2000)
    ap_build.add_argument("--overwrite", action="store_true")
    ap_build.add_argument("--num_workers", type=int, default=0)

    ap_sw = sub.add_parser("dataset_sweep", help="Sweep over dataset combinations, train CMAOTN-NPZ model, and aggregate results")
    ap_sw.add_argument("--labels_csv", type=str, required=True)
    ap_sw.add_argument("--runs_root", type=str, required=True)
    ap_sw.add_argument("--run_name", type=str, required=True)

    ap_sw.add_argument("--max_points", type=int, default=2000)
    ap_sw.add_argument("--kfold", type=int, default=5)
    ap_sw.add_argument("--seed", type=int, default=42)
    ap_sw.add_argument("--img_size", type=int, default=224)
    ap_sw.add_argument("--batch_size", type=int, default=24)
    ap_sw.add_argument("--epoch", "--epochs", dest="epochs", type=int, default=10)
    ap_sw.add_argument("--lr", type=float, default=5e-4)
    ap_sw.add_argument("--weight_decay", type=float, default=1e-4)
    ap_sw.add_argument("--lr_scheduler", choices=["none", "cosine", "cosine_warmup", "onecycle", "plateau"], default="cosine_warmup")
    ap_sw.add_argument("--lr_warmup_ratio", type=float, default=0.1)
    ap_sw.add_argument("--lr_min", type=float, default=5e-6)
    ap_sw.add_argument("--lr_plateau_factor", type=float, default=0.5)
    ap_sw.add_argument("--lr_plateau_patience", type=int, default=2)
    ap_sw.add_argument("--backbone_lr_mult", type=float, default=1.0)
    ap_sw.add_argument("--num_workers", type=int, default=4)
    ap_sw.add_argument("--fusion", choices=["concat", "stack"], default="concat")
    ap_sw.add_argument("--val_ratio", type=float, default=0.2)
    ap_sw.add_argument("--metric_for_best", choices=["macro_f1", "acc"], default="macro_f1")
    ap_sw.add_argument("--confidence_gap_threshold", type=float, default=0.3)

    ap_sw.add_argument("--chan_proj", choices=["mlp", "linear"], default="mlp")
    ap_sw.add_argument("--chan_hidden", type=int, default=0)
    ap_sw.add_argument("--chan_norm", choices=["group", "none"], default="group")
    ap_sw.add_argument("--chan_dropout", type=float, default=0.0)

    ap_sw.add_argument("--npz_norm", choices=["log1p_global", "per_sample_minmax", "none"], default="log1p_global")
    ap_sw.add_argument("--npz_global_max_log", type=float, default=10.0)

    ap_sw.add_argument("--finetune_backbone", dest="finetune_backbone", action="store_true")
    ap_sw.add_argument("--freeze_backbone", dest="finetune_backbone", action="store_false")
    ap_sw.set_defaults(finetune_backbone=True)
    ap_sw.add_argument("--use_class_weights", action="store_true")

    ap_sw.add_argument("--use_lora", action="store_true")
    ap_sw.add_argument("--lora_backend", choices=["auto", "peft", "custom"], default="auto")
    ap_sw.add_argument("--lora_r", type=int, default=8)
    ap_sw.add_argument("--lora_alpha", type=float, default=16.0)
    ap_sw.add_argument("--lora_dropout", type=float, default=0.0)
    ap_sw.add_argument("--lora_targets", type=str, default="pwconv,fc,qkv,proj")

    ap_sw.add_argument("--allow_fallback_backbone", action="store_true")
    ap_sw.add_argument("--hf_model_id", type=str, default="facebook/dinov3-convnext-tiny-pretrain-lvd1689m")
    ap_sw.add_argument("--hf_local_only", action="store_true")

    ap_sw.add_argument("--min_samples", type=int, default=21)

    args = ap.parse_args()

    if hasattr(args, "labels_csv"):
        args.labels_csv = ref.remap_known_root(args.labels_csv)
    if hasattr(args, "runs_root"):
        args.runs_root = ref.remap_known_root(args.runs_root)

    if args.cmd == "build_cmaotn":
        spans = ref._parse_spans(args.spans)
        return ref.build_cmaotn_images_from_csv(
            labels_csv=args.labels_csv,
            out_size=int(args.out_size),
            m=int(args.m),
            tau=int(args.tau),
            spans=spans,
            max_points=int(args.max_points),
            overwrite=bool(args.overwrite),
            num_workers=int(args.num_workers),
        )

    if args.cmd == "dataset_sweep":
        return run_dataset_sweep(**vars(args))

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
