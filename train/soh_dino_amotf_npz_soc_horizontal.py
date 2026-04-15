#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path


# python /mnt/sdb/THX/Battery_THX_HP_P9000/Battery/dataset/Tao/Battery_Archive/scripts/soh_dino_cmaotn_npz_soc_horizontal.py train \
#   --labels_csv /mnt/sdb/THX/Battery_THX_HP_P9000/no_title_outputs/features/soh_classification_results.csv \
#   --runs_root /mnt/sdb/THX/Battery_THX_HP_P9000/no_title_outputs/soh_cmaotn_dino_runs \
#   --run_name exp_full_soc_horizontal_npz \
#   --input_mode cmaotn_npz \
#   --finetune_backbone \
#   --lr 5e-4 \
#   --npz_norm log1p_global \
#   --npz_global_max_log 10.0 \
#   --use_class_weights \
#   --backbone_lr_mult 0.1 \
#   --lr_scheduler cosine_warmup \
#   --lr_warmup_ratio 0.1 \
#   --lr_min 1e-6 \
#   --epochs 50

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import soh_dino_cmaotn_npz_partial_cycles_partial_sweep_soc_horizontal as ref


def main() -> int:
    ap = argparse.ArgumentParser(
        description="CMAOTN image build + DINOv3 SOH classification (SOC-horizontal, full-data only)."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_build = sub.add_parser("build_cmaotn", help="Build CMAOTN images (PNG/NPZ) under each sample dir/cmaotn/")
    ap_build.add_argument("--labels_csv", type=str, default=ref._default_labels_csv())
    ap_build.add_argument("--out_size", type=int, default=0)
    ap_build.add_argument("--m", type=int, default=5)
    ap_build.add_argument("--tau", type=int, default=1)
    ap_build.add_argument("--spans", type=str, default="1,2,4,8")
    ap_build.add_argument("--max_points", type=int, default=2000)
    ap_build.add_argument("--overwrite", action="store_true")
    ap_build.add_argument("--num_workers", type=int, default=0)

    ap_train = sub.add_parser("train", help="Train classifier on full data")
    ap_train.add_argument("--labels_csv", type=str, default=ref._default_labels_csv())
    ap_train.add_argument("--runs_root", type=str, default=ref._default_runs_root())
    ap_train.add_argument("--run_name", type=str, default="")
    ap_train.add_argument("--input_mode", choices=["cmaotn", "cmaotn_npz", "png"], default="cmaotn_npz")
    ap_train.add_argument("--max_points", type=int, default=2000)
    ap_train.add_argument("--kfold", type=int, default=5)
    ap_train.add_argument("--split_indices_json", type=str, default="")
    ap_train.add_argument("--exclude_samples_txt", type=str, default=ref._default_exclude_samples_txt())
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

    ap_all = sub.add_parser("run_all", help="Build CMAOTN then train CMAOTN and PNG baseline under one run dir")
    ap_all.add_argument("--labels_csv", type=str, default=ref._default_labels_csv())
    ap_all.add_argument("--runs_root", type=str, default=ref._default_runs_root())
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
    ap_all.add_argument("--exclude_samples_txt", type=str, default=ref._default_exclude_samples_txt())
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
    if hasattr(args, "labels_csv"):
        args.labels_csv = ref.remap_known_root(args.labels_csv)
    if hasattr(args, "runs_root"):
        args.runs_root = ref.remap_known_root(args.runs_root)
    if hasattr(args, "split_indices_json"):
        args.split_indices_json = ref.remap_known_root(getattr(args, "split_indices_json", ""))
    if hasattr(args, "exclude_samples_txt"):
        args.exclude_samples_txt = ref.remap_known_root(getattr(args, "exclude_samples_txt", ""))

    if args.cmd == "build_cmaotn":
        spans = ref._parse_spans(args.spans)
        return ref.build_cmaotn_images_from_csv(
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
        ref._ensure_sklearn()
        return ref.train_dino_soh_classifier(**vars(args))

    if args.cmd == "run_all":
        ref._ensure_sklearn()
        spans = ref._parse_spans(args.spans)
        run_name = str(args.run_name).strip() or ref._now_run_name()
        run_root = Path(args.runs_root) / run_name
        ref._ensure_dir(run_root)
        ref._setup_logging(run_root / "run_all.log")

        ref.LOGGER.info("run_all run_name=%s", run_name)
        ref.LOGGER.info("step=build_cmaotn")
        rc = ref.build_cmaotn_images_from_csv(
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
            ref.LOGGER.error("build_cmaotn failed rc=%d", rc)
            return int(rc)

        keep, stats = ref._collect_intersection_original_paths(args.labels_csv, max_points=int(args.max_points))
        ref.LOGGER.info("intersection_stats=%s", stats)
        if int(stats.get("keep", 0)) < 20:
            ref.LOGGER.error("Too few intersection samples: %s", stats)
            return 2

        subset_csv = run_root / "labels_intersection.csv"
        ref._write_labels_subset(args.labels_csv, keep, subset_csv)

        ref.LOGGER.info("step=train_cmaotn_npz")
        rc1 = ref.train_dino_soh_classifier(
            labels_csv=str(subset_csv),
            runs_root=str(args.runs_root),
            run_name=run_name,
            input_mode="cmaotn_npz",
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

        ref.LOGGER.info("step=train_png")
        rc2 = ref.train_dino_soh_classifier(
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

        ref._compare_runs(run_root)
        ref.LOGGER.info("run_all finished. compare_table=%s", str(run_root / "compare_table.csv"))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
