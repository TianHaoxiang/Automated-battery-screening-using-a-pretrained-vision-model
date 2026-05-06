#!/usr/bin/env python
import argparse
import csv
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_DIR = PROJECT_ROOT / "train"
EXTRACT_DIR = PROJECT_ROOT / "extract_feature"

DATASET_WRAPPERS: Dict[str, str] = {
    "CALCE": "extract_CALCE_feature.py",
    "HNEI": "extract_HNEI_feature.py",
    "Oxford": "extract_Oxford_feature.py",
    "SNL_LFP": "extract_SNL_LFP_feature.py",
    "SNL_NCA": "extract_SNL_NCA_feature.py",
    "SNL_NMC": "extract_SNL_NMC_feature.py",
    "UL_Purdue": "extract_UL_Purdue_feature.py",
}


def _default_project_root() -> str:
    return "/mnt/sdb/THX/Battery_THX_HP_P9000"


def _default_portable_labels_csv() -> str:
    return str(PROJECT_ROOT / "data" / "soh_classification_results_portable.csv")


def _default_source_labels_csv() -> str:
    return os.path.join(_default_project_root(), "no_title_outputs", "features", "soh_classification_results.csv")


def _default_labels_csv() -> str:
    portable = _default_portable_labels_csv()
    if os.path.exists(portable):
        return portable
    return _default_source_labels_csv()


def _default_runs_root() -> str:
    return os.path.join(_default_project_root(), "no_title_outputs", "soh_amotf_dino_runs")


def _default_exclude_samples_txt() -> str:
    return os.path.join(_default_project_root(), "no_title_outputs", "exclude_samples.txt")


def _default_stability_split_indices_json() -> str:
    env = str(os.environ.get("BATTERY_SOH_STABILITY_SPLIT_JSON", "")).strip()
    if env:
        return env
    bundled = PROJECT_ROOT / "data" / "stability_fold1_split_indices.json"
    if bundled.is_file():
        return str(bundled)
    return os.path.join(_default_runs_root(), "run_20260505_011002", "amotf_npz", "fold_1", "split_indices.json")


def _default_archive_root() -> str:
    return os.path.join(
        _default_project_root(),
        "Battery",
        "dataset",
        "Tao",
        "Battery_Archive",
    )


def _default_feature_out_dir() -> str:
    return os.path.join(_default_archive_root(), "Battery_Archive", "outputs_analysis", "features")


def _normalize_slashes(p: str) -> str:
    return str(p).replace("\\", "/")


def _strip_known_data_root(p: str) -> str:
    s = _normalize_slashes(p).strip()
    for prefix in [
        _default_project_root(),
        "/media/haoxiang/THX_HP_P900",
        "F:/",
        "F:",
    ]:
        q = _normalize_slashes(prefix).rstrip("/")
        if s.lower().startswith((q + "/").lower()):
            return s[len(q) + 1 :]
        if s.lower() == q.lower():
            return ""
    return s


def _write_portable_labels_csv(*, in_labels_csv: str, out_labels_csv: str) -> None:
    src = Path(in_labels_csv)
    if not src.exists():
        raise FileNotFoundError(str(src))
    rows: List[dict] = []
    with open(src, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            row = dict(row)
            row["original_path"] = _strip_known_data_root(str(row.get("original_path", "")).strip())
            rows.append(row)
    dst = Path(out_labels_csv)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run(cmd: List[str], *, dry_run: bool) -> int:
    print("[launcher] cwd=", str(PROJECT_ROOT))
    print("[launcher] cmd=", " ".join(cmd))
    if dry_run:
        return 0
    rc = subprocess.call(cmd, cwd=str(PROJECT_ROOT))
    if rc < 0:
        sig_num = -int(rc)
        try:
            sig_name = signal.Signals(sig_num).name
        except Exception:
            sig_name = f"SIG{sig_num}"
        print(f"[launcher] child terminated by signal {sig_num} ({sig_name})")
        return 128 + sig_num
    return int(rc)


def _tail_epoch_lines(text: str, *, n: int = 3) -> List[str]:
    lines = [ln for ln in text.splitlines() if " fold=" in ln and " epoch=" in ln and "train_loss=" in ln]
    return lines[-n:] if lines else []


def _last_epoch_from_log(text: str) -> Optional[int]:
    best: Optional[int] = None
    for ln in text.splitlines():
        if " epoch=" not in ln or "train_loss=" not in ln:
            continue
        m = re.search(r"epoch=(\d+)", ln)
        if m:
            best = int(m.group(1))
    return best


def _monitor_train_log(*, log_path: Path, proc: subprocess.Popen, interval_sec: int, expect_epochs: int) -> None:
    last_ep: Optional[int] = None
    same_count = 0
    while proc.poll() is None:
        time.sleep(max(5, int(interval_sec)))
        try:
            raw = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            print(f"[monitor] cannot read {log_path}: {e}")
            continue
        ep = _last_epoch_from_log(raw)
        sz = log_path.stat().st_size if log_path.exists() else 0
        tail = _tail_epoch_lines(raw, n=2)
        if ep is None:
            print(f"[monitor] waiting for first epoch line… log_bytes={sz}")
            continue
        if last_ep is not None and ep == last_ep:
            same_count += 1
            print(
                f"[monitor] no new epoch yet (epoch={ep}, repeats={same_count}, log_bytes={sz}). "
                f"If repeats grow while GPU is idle, check NFS/dataloader/OOM."
            )
        else:
            if last_ep is not None and ep < last_ep:
                print(f"[monitor] warn: epoch decreased {last_ep} -> {ep}")
            same_count = 0
            last_ep = ep
            print(f"[monitor] progress epoch={ep}/{expect_epochs - 1} log_bytes={sz}")
        for ln in tail:
            print(f"[monitor]   {ln}")


def _train_python_executable() -> str:
    return str(os.environ.get("BATTERY_SOH_TRAIN_PYTHON", "") or "").strip() or sys.executable


def _python_can_import_torch(python_exe: str) -> bool:
    try:
        r = subprocess.run(
            [python_exe, "-c", "import torch"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        return int(r.returncode) == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _stability_fold1_train_cmd(
    *,
    labels_csv: str,
    runs_root: str,
    run_name: str,
    split_indices_json: str,
    exclude_samples_txt: str,
    seed: int,
    epochs: int,
    quick_judge: bool,
    quick_judge_train_batches: int,
    quick_judge_eval_batches: int,
) -> List[str]:
    cmd: List[str] = [
        _train_python_executable(),
        "-u",
        str(TRAIN_DIR / "soh_dino_amotf_npz_soc_horizontal.py"),
        "train",
        "--labels_csv",
        str(labels_csv),
        "--runs_root",
        str(runs_root),
        "--run_name",
        str(run_name),
        "--input_mode",
        "amotf_npz",
        "--split_indices_json",
        str(split_indices_json),
        "--seed",
        str(int(seed)),
        "--img_size",
        "224",
        "--batch_size",
        "24",
        "--epochs",
        str(int(epochs)),
        "--lr",
        "5e-4",
        "--weight_decay",
        "1e-4",
        "--lr_scheduler",
        "cosine_warmup",
        "--lr_warmup_ratio",
        "0.1",
        "--lr_min",
        "1e-6",
        "--lr_plateau_factor",
        "0.5",
        "--lr_plateau_patience",
        "2",
        "--backbone_lr_mult",
        "0.1",
        "--num_workers",
        "0",
        "--fusion",
        "concat",
        "--val_ratio",
        "0.2",
        "--metric_for_best",
        "macro_f1",
        "--confidence_gap_threshold",
        "0.3",
        "--chan_proj",
        "mlp",
        "--chan_hidden",
        "0",
        "--chan_norm",
        "group",
        "--chan_dropout",
        "0.0",
        "--npz_norm",
        "log1p_global",
        "--npz_global_max_log",
        "10.0",
        "--finetune_backbone",
        "--use_class_weights",
        "--hf_model_id",
        "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
        "--hf_local_only",
        "--stability_mode",
    ]
    if exclude_samples_txt and Path(exclude_samples_txt).is_file():
        cmd.extend(["--exclude_samples_txt", str(exclude_samples_txt)])
    if bool(quick_judge):
        cmd.extend(
            [
                "--quick_judge",
                "--quick_judge_train_batches",
                str(max(1, int(quick_judge_train_batches))),
                "--quick_judge_eval_batches",
                str(max(1, int(quick_judge_eval_batches))),
            ]
        )
    return cmd


def _append_missing_option(cmd: List[str], extra: List[str], flag: str, value: str) -> None:
    if not any(arg == flag or arg.startswith(f"{flag}=") for arg in extra):
        cmd.extend([flag, value])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run battery-soh-dino workflows from the repository root with one command.",
        epilog=(
            "Examples:\n"
            f"  python run_battery_soh_dino.py extract_features --datasets CALCE --max_cells 2\n"
            f"  python run_battery_soh_dino.py build_amotf --labels_csv {_default_labels_csv()}\n"
            f"  python run_battery_soh_dino.py run_all --labels_csv {_default_labels_csv()} --runs_root {_default_runs_root()} --run_name exp_full_finetune_run_all --finetune_backbone --lr 5e-4 --npz_norm log1p_global --npz_global_max_log 10.0 --use_class_weights --backbone_lr_mult 0.1 --lr_scheduler cosine_warmup --lr_warmup_ratio 0.1 --lr_min 1e-6 --epochs 50\n"
            f"  python run_battery_soh_dino.py dataset_sweep --labels_csv {_default_labels_csv()} --runs_root {_default_runs_root()} --run_name exp_dataset_sweep --finetune_backbone --lr 5e-4 --epochs 50\n"
            f"  python run_battery_soh_dino.py --dry_run stability_fold1\n"
            f"  conda activate dinov3 && python run_battery_soh_dino.py stability_fold1\n"
            f"  conda activate dinov3 && python run_battery_soh_dino.py stability_fold1 --quick_judge\n"
            f"  BATTERY_SOH_TRAIN_PYTHON=/path/to/dinov3/bin/python python run_battery_soh_dino.py stability_fold1 --epochs 2"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--dry_run", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("extract_features", help="Run extract_feature/extract_Battery_Archive_feature.py")

    ap_extract_dataset = sub.add_parser("extract_dataset", help="Run one dataset-specific extract_feature wrapper")
    ap_extract_dataset.add_argument("--dataset", choices=sorted(DATASET_WRAPPERS.keys()), required=True)

    sub.add_parser("build_amotf", help="Run full-data build_amotf")
    sub.add_parser("train", help="Run full-data train")
    sub.add_parser("run_all", help="Run full-data run_all")
    sub.add_parser("dataset_build_amotf", help="Run dataset-sweep build_amotf")
    sub.add_parser("dataset_sweep", help="Run dataset combination sweep")
    ap_portable = sub.add_parser("make_portable_labels", help="Create a repo-local portable labels CSV with relative original_path values")
    ap_portable.add_argument("--in_labels_csv", type=str, default=_default_source_labels_csv())
    ap_portable.add_argument("--out_labels_csv", type=str, default=_default_portable_labels_csv())
    sub.add_parser("print_defaults", help="Print default paths used in the README examples")

    ap_stab = sub.add_parser(
        "stability_fold1",
        help=(
            "Single-fold AMOTF NPZ train on a fixed split_indices.json (default fold1 split), "
            "50 epochs, same hyperparameters as the pre-refactor hardened stability run, optional log monitor."
        ),
    )
    ap_stab.add_argument("--labels_csv", type=str, default=_default_labels_csv())
    ap_stab.add_argument("--runs_root", type=str, default=_default_runs_root())
    ap_stab.add_argument(
        "--run_name",
        type=str,
        default="",
        help="If empty, uses stability_fold1_<YYYYMMDD_HHMMSS> under runs_root/amotf_npz/train.log",
    )
    ap_stab.add_argument(
        "--split_indices_json",
        type=str,
        default="",
        help="If empty, uses BATTERY_SOH_STABILITY_SPLIT_JSON or run_20260505_011002/.../fold_1/split_indices.json",
    )
    ap_stab.add_argument(
        "--exclude_samples_txt",
        type=str,
        default=_default_exclude_samples_txt(),
        help="Passed only if this file exists.",
    )
    ap_stab.add_argument("--seed", type=int, default=42)
    ap_stab.add_argument("--epochs", type=int, default=50)
    ap_stab.add_argument("--quick_judge", action="store_true", help="Run a real first-batch/eval/checkpoint probe and exit early.")
    ap_stab.add_argument("--quick_judge_train_batches", type=int, default=1)
    ap_stab.add_argument("--quick_judge_eval_batches", type=int, default=1)
    ap_stab.add_argument(
        "--monitor_interval_sec",
        type=int,
        default=120,
        help="Background monitor polls train.log on this interval (0 disables).",
    )
    ap_stab.add_argument("--no_monitor", action="store_true", help="Disable background train.log monitor thread.")
    ap_stab.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the resolved train command and exit (no GPU, no torch import check).",
    )

    args, extra = parser.parse_known_args()

    if args.cmd == "print_defaults":
        print("PROJECT_ROOT=", str(PROJECT_ROOT))
        print("DEFAULT_ARCHIVE_ROOT=", _default_archive_root())
        print("DEFAULT_FEATURE_OUT_DIR=", _default_feature_out_dir())
        print("DEFAULT_SOURCE_LABELS_CSV=", _default_source_labels_csv())
        print("DEFAULT_PORTABLE_LABELS_CSV=", _default_portable_labels_csv())
        print("DEFAULT_LABELS_CSV=", _default_labels_csv())
        print("DEFAULT_RUNS_ROOT=", _default_runs_root())
        return 0

    if args.cmd == "make_portable_labels":
        _write_portable_labels_csv(in_labels_csv=str(args.in_labels_csv), out_labels_csv=str(args.out_labels_csv))
        print("PORTABLE_LABELS_CSV=", str(Path(args.out_labels_csv).resolve()))
        return 0

    if args.cmd == "extract_features":
        cmd = [sys.executable, str(EXTRACT_DIR / "extract_Battery_Archive_feature.py")]
        _append_missing_option(cmd, extra, "--root_dir", _default_archive_root())
        _append_missing_option(cmd, extra, "--out_dir", _default_feature_out_dir())
        cmd.extend(extra)
        return _run(cmd, dry_run=bool(args.dry_run))

    if args.cmd == "extract_dataset":
        wrapper = DATASET_WRAPPERS[str(args.dataset)]
        cmd = [sys.executable, str(EXTRACT_DIR / wrapper)]
        _append_missing_option(cmd, extra, "--root_dir", _default_archive_root())
        _append_missing_option(cmd, extra, "--out_dir", _default_feature_out_dir())
        cmd.extend(extra)
        return _run(cmd, dry_run=bool(args.dry_run))

    if args.cmd == "build_amotf":
        cmd = [sys.executable, str(TRAIN_DIR / "soh_dino_amotf_npz_soc_horizontal.py"), "build_amotf"]
        _append_missing_option(cmd, extra, "--labels_csv", _default_labels_csv())
        cmd.extend(extra)
        return _run(cmd, dry_run=bool(args.dry_run))

    if args.cmd == "train":
        cmd = [sys.executable, str(TRAIN_DIR / "soh_dino_amotf_npz_soc_horizontal.py"), "train"]
        _append_missing_option(cmd, extra, "--labels_csv", _default_labels_csv())
        _append_missing_option(cmd, extra, "--runs_root", _default_runs_root())
        cmd.extend(extra)
        return _run(cmd, dry_run=bool(args.dry_run))

    if args.cmd == "run_all":
        cmd = [sys.executable, str(TRAIN_DIR / "soh_dino_amotf_npz_soc_horizontal.py"), "run_all"]
        _append_missing_option(cmd, extra, "--labels_csv", _default_labels_csv())
        _append_missing_option(cmd, extra, "--runs_root", _default_runs_root())
        cmd.extend(extra)
        return _run(cmd, dry_run=bool(args.dry_run))

    if args.cmd == "dataset_build_amotf":
        cmd = [sys.executable, str(TRAIN_DIR / "soh_dino_amotf_npz_dataset_cycles_all_classes_soc_horizontal.py"), "build_amotf"]
        _append_missing_option(cmd, extra, "--labels_csv", _default_labels_csv())
        cmd.extend(extra)
        return _run(cmd, dry_run=bool(args.dry_run))

    if args.cmd == "dataset_sweep":
        cmd = [sys.executable, str(TRAIN_DIR / "soh_dino_amotf_npz_dataset_cycles_all_classes_soc_horizontal.py"), "dataset_sweep"]
        _append_missing_option(cmd, extra, "--labels_csv", _default_labels_csv())
        _append_missing_option(cmd, extra, "--runs_root", _default_runs_root())
        cmd.extend(extra)
        return _run(cmd, dry_run=bool(args.dry_run))

    if args.cmd == "stability_fold1":
        dry_here = bool(getattr(args, "dry_run", False)) or ("--dry_run" in sys.argv)
        split_json = str(args.split_indices_json or "").strip() or _default_stability_split_indices_json()
        if not Path(split_json).is_file():
            print(
                "[stability_fold1] ERROR: split_indices_json not found:\n  ",
                split_json,
                "\n  Set --split_indices_json or env BATTERY_SOH_STABILITY_SPLIT_JSON to a valid fold_1/split_indices.json",
                sep="",
            )
            return 2
        if not Path(args.labels_csv).is_file():
            print("[stability_fold1] ERROR: labels_csv not found:", args.labels_csv)
            return 2
        run_name = str(args.run_name).strip()
        if not run_name:
            run_name = "stability_fold1_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        cmd = _stability_fold1_train_cmd(
            labels_csv=str(args.labels_csv),
            runs_root=str(args.runs_root),
            run_name=run_name,
            split_indices_json=split_json,
            exclude_samples_txt=str(args.exclude_samples_txt),
            seed=int(args.seed),
            epochs=int(args.epochs),
            quick_judge=bool(args.quick_judge),
            quick_judge_train_batches=int(args.quick_judge_train_batches),
            quick_judge_eval_batches=int(args.quick_judge_eval_batches),
        )
        log_path = Path(args.runs_root) / run_name / "amotf_npz" / "train.log"
        quick_judge_report = Path(args.runs_root) / run_name / "amotf_npz" / "fold_1" / "quick_judge_report.json"
        print("[stability_fold1] run_name=", run_name)
        print("[stability_fold1] train.log ->", str(log_path))
        if bool(args.quick_judge):
            print("[stability_fold1] quick_judge_report ->", str(quick_judge_report))
        print("[stability_fold1] cwd=", str(PROJECT_ROOT))
        print("[stability_fold1] cmd=", " ".join(cmd))
        print("[stability_fold1] second terminal (optional): tail -f", str(log_path))
        if dry_here:
            return 0
        py_exe = _train_python_executable()
        if not _python_can_import_torch(py_exe):
            print(
                "[stability_fold1] ERROR: selected Python cannot import torch:\n  ",
                py_exe,
                "\n  Fix: conda activate dinov3  (or your env with torch), then re-run;\n"
                "  or: BATTERY_SOH_TRAIN_PYTHON=/path/to/that/python python run_battery_soh_dino.py stability_fold1 …",
                sep="",
            )
            return 2
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env)
        mon: Optional[threading.Thread] = None
        interval = int(args.monitor_interval_sec)
        if (not bool(args.no_monitor)) and interval > 0:
            mon = threading.Thread(
                target=_monitor_train_log,
                kwargs={
                    "log_path": log_path,
                    "proc": proc,
                    "interval_sec": interval,
                    "expect_epochs": int(args.epochs),
                },
                daemon=True,
            )
            mon.start()
        rc = int(proc.wait())
        if rc != 0:
            print(f"[stability_fold1] training exited with code {rc}")
            return rc
        if bool(args.quick_judge):
            print("[stability_fold1] quick_judge finished. Check:")
            print("  ", str(log_path))
            print("  ", str(quick_judge_report), "exists=", quick_judge_report.is_file())
            if not quick_judge_report.is_file():
                print("[stability_fold1] WARN: quick_judge_report.json missing; inspect train.log for errors.")
                return 3
            try:
                raw = quick_judge_report.read_text(encoding="utf-8", errors="replace")
                print(raw)
                parsed = json.loads(raw)
            except OSError as e:
                print(f"[stability_fold1] WARN: cannot read quick_judge_report.json: {e}")
                return 3
            except json.JSONDecodeError as e:
                print(f"[stability_fold1] WARN: invalid quick_judge_report.json: {e}")
                return 3
            status = str(parsed.get("status", "")).strip().lower()
            if status != "pass":
                print(f"[stability_fold1] quick_judge status is not pass: {status or 'missing'}")
                return 3
            return 0
        best_pt = Path(args.runs_root) / run_name / "amotf_npz" / "fold_1" / "best.pt"
        fold_csv = Path(args.runs_root) / run_name / "amotf_npz" / "fold_metrics.csv"
        print("[stability_fold1] done. Check:")
        print("  ", str(log_path))
        print("  ", str(best_pt), "exists=", best_pt.is_file())
        print("  ", str(fold_csv), "exists=", fold_csv.is_file())
        if not best_pt.is_file():
            print("[stability_fold1] WARN: best.pt missing; inspect train.log for errors.")
            return 3
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
