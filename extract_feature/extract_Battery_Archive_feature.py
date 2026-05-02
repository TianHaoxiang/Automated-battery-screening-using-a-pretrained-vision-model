import argparse
import os

import pandas as pd

import battery_archive_feature_lib as lib


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Battery Archive local dataset: single-observation (last complete cycle) feature extraction & visualization"
    )
    parser.add_argument("--root_dir", default=r"/mnt/sdb/THX/Battery_THX_HP_P9000/Battery/dataset/Tao/Battery_Archive/Battery_Archive")
    parser.add_argument("--out_dir", default=r"/mnt/sdb/THX/Battery_THX_HP_P9000/Battery/dataset/Tao/Battery_Archive/Battery_Archive/outputs_analysis/features")
    parser.add_argument("--mode", default="all_cycles", choices=["last_only", "all_cycles"])
    parser.add_argument("--min_points", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_cells", type=int, default=None)
    parser.add_argument("--debug", action="store_true")

    parser.add_argument(
        "--summarize_only",
        action="store_true",
        help="Only aggregate failed/skipped reason stats from existing index_samples.csv files; do not re-extract samples.",
    )
    parser.add_argument(
        "--summary_in_dir",
        default=None,
        help="When --summarize_only is set, read index_samples.csv from this directory (defaults to --out_dir).",
    )

    # Optional filtering
    parser.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated dataset folder names to process (e.g. 'UL-Purdue,CALCE'). Default: all.",
    )

    args = parser.parse_args()
    lib.setup_logging(args.debug)

    root_dir = str(args.root_dir)
    if (not os.path.isdir(os.path.join(root_dir, "Battery_Archive"))) and (os.path.basename(root_dir) == "Battery_Archive"):
        root_dir = os.path.dirname(root_dir)
    args.root_dir = root_dir

    dataset_folders = lib.discover_dataset_folders(args.root_dir)

    selected = None
    if args.datasets:
        selected = {s.strip() for s in str(args.datasets).split(",") if s.strip()}
        dataset_folders = [d for d in dataset_folders if d in selected]

    # Record dataset_info.xlsx columns in global summary
    dataset_info_path = os.path.join(args.root_dir, "Battery_Archive", "dataset_info.xlsx")
    try:
        dataset_info_df, _ = lib.read_dataset_info_xlsx(dataset_info_path, debug=args.debug)
        dataset_info_cols = list(dataset_info_df.columns)
    except Exception as e:
        lib.LOGGER.warning("Failed to read dataset_info.xlsx: %s", e)
        dataset_info_cols = None

    all_index_rows = []
    if not bool(args.summarize_only):
        for folder in dataset_folders:
            try:
                lib.LOGGER.info("Processing dataset folder: %s", folder)
                lib.run_dataset(
                    root_dir=args.root_dir,
                    dataset_folder_name=folder,
                    out_dir=args.out_dir,
                    mode=args.mode,
                    min_points=args.min_points,
                    overwrite=bool(args.overwrite),
                    max_cells=args.max_cells,
                    debug=args.debug,
                )

                out_dataset_dir = os.path.join(args.out_dir, lib.safe_filename(folder))
                idx_csv = os.path.join(out_dataset_dir, "index_samples.csv")
                if os.path.exists(idx_csv):
                    df = pd.read_csv(idx_csv)
                    all_index_rows.append(df)
            except Exception:
                lib.LOGGER.exception("Dataset failed: %s", folder)
    else:
        in_dir = str(args.summary_in_dir).strip() if args.summary_in_dir else str(args.out_dir)
        in_dir = str(in_dir)
        lib.LOGGER.info("summarize_only=1 summary_in_dir=%s", in_dir)
        for folder in dataset_folders:
            out_dataset_dir = os.path.join(in_dir, lib.safe_filename(folder))
            idx_csv = os.path.join(out_dataset_dir, "index_samples.csv")
            if os.path.exists(idx_csv):
                try:
                    df = pd.read_csv(idx_csv)
                    all_index_rows.append(df)
                except Exception:
                    lib.LOGGER.exception("Failed to read index_samples.csv: %s", idx_csv)

        if not all_index_rows:
            merged_existing = os.path.join(in_dir, "index_samples.csv")
            if os.path.exists(merged_existing):
                try:
                    df = pd.read_csv(merged_existing)
                    all_index_rows.append(df)
                except Exception:
                    lib.LOGGER.exception("Failed to read merged index_samples.csv: %s", merged_existing)

    if all_index_rows:
        merged = pd.concat(all_index_rows, axis=0, ignore_index=True)
    else:
        merged = pd.DataFrame([])

    lib.ensure_dir(args.out_dir)
    merged_csv = os.path.join(args.out_dir, "index_samples.csv")
    merged_xlsx = os.path.join(args.out_dir, "index_samples.xlsx")

    def _status_distribution_by_dataset(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame([])
        if "dataset_name" not in df.columns or "status" not in df.columns:
            return pd.DataFrame([])
        tmp = df.copy()
        tmp["dataset_name"] = tmp["dataset_name"].fillna("").astype(str)
        tmp["status"] = tmp["status"].fillna("").astype(str)
        g = tmp.groupby(["dataset_name", "status"]).size().reset_index(name="count")
        piv = g.pivot_table(index="dataset_name", columns="status", values="count", aggfunc="sum", fill_value=0)
        piv = piv.reset_index()
        num_cols = [c for c in piv.columns if c != "dataset_name"]
        if num_cols:
            piv["total"] = piv[num_cols].sum(axis=1)
        return piv

    def _top_reasons_by_dataset(df: pd.DataFrame, *, status: str, reason_col: str, topn: int = 10) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame([])
        if "dataset_name" not in df.columns or "status" not in df.columns or reason_col not in df.columns:
            return pd.DataFrame([])
        tmp = df.loc[df["status"].astype(str) == str(status)].copy()
        if tmp.empty:
            return pd.DataFrame([])
        tmp["dataset_name"] = tmp["dataset_name"].fillna("").astype(str)
        tmp[reason_col] = tmp[reason_col].fillna("").astype(str)
        tmp = tmp.loc[tmp[reason_col].astype(str).str.len() > 0]
        if tmp.empty:
            return pd.DataFrame([])
        g = tmp.groupby(["dataset_name", reason_col]).size().reset_index(name="count")
        g = g.sort_values(["dataset_name", "count", reason_col], ascending=[True, False, True])
        g["rank"] = g.groupby("dataset_name").cumcount() + 1
        g = g.loc[g["rank"] <= int(topn)].drop(columns=["rank"])
        return g

    def _top_reasons_overall(df: pd.DataFrame, *, status: str, reason_col: str, topn: int = 30) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame([])
        if "status" not in df.columns or reason_col not in df.columns:
            return pd.DataFrame([])
        tmp = df.loc[df["status"].astype(str) == str(status)].copy()
        if tmp.empty:
            return pd.DataFrame([])
        tmp[reason_col] = tmp[reason_col].fillna("").astype(str)
        tmp = tmp.loc[tmp[reason_col].astype(str).str.len() > 0]
        if tmp.empty:
            return pd.DataFrame([])
        g = tmp[reason_col].value_counts().reset_index()
        g.columns = [reason_col, "count"]
        g = g.head(int(topn))
        return g

    status_by_dataset = _status_distribution_by_dataset(merged)
    failed_reasons_by_dataset = _top_reasons_by_dataset(merged, status="failed", reason_col="fail_reason", topn=10)
    skipped_reasons_by_dataset = _top_reasons_by_dataset(merged, status="skipped", reason_col="completeness_reason", topn=10)
    failed_reasons_overall = _top_reasons_overall(merged, status="failed", reason_col="fail_reason", topn=30)
    skipped_reasons_overall = _top_reasons_overall(merged, status="skipped", reason_col="completeness_reason", topn=30)

    status_csv = os.path.join(args.out_dir, "status_by_dataset.csv")
    failed_csv = os.path.join(args.out_dir, "failed_reasons_by_dataset.csv")
    skipped_csv = os.path.join(args.out_dir, "skipped_reasons_by_dataset.csv")
    failed_overall_csv = os.path.join(args.out_dir, "failed_reasons_overall.csv")
    skipped_overall_csv = os.path.join(args.out_dir, "skipped_reasons_overall.csv")

    merged.to_csv(merged_csv, index=False)
    with pd.ExcelWriter(merged_xlsx, engine="openpyxl") as writer:
        merged.to_excel(writer, index=False, sheet_name="samples")
        status_by_dataset.to_excel(writer, index=False, sheet_name="status_by_dataset")
        failed_reasons_by_dataset.to_excel(writer, index=False, sheet_name="failed_reasons")
        skipped_reasons_by_dataset.to_excel(writer, index=False, sheet_name="skipped_reasons")
        failed_reasons_overall.to_excel(writer, index=False, sheet_name="failed_overall")
        skipped_reasons_overall.to_excel(writer, index=False, sheet_name="skipped_overall")

    status_by_dataset.to_csv(status_csv, index=False)
    failed_reasons_by_dataset.to_csv(failed_csv, index=False)
    skipped_reasons_by_dataset.to_csv(skipped_csv, index=False)
    failed_reasons_overall.to_csv(failed_overall_csv, index=False)
    skipped_reasons_overall.to_csv(skipped_overall_csv, index=False)

    summary = {
        "root_dir": args.root_dir,
        "out_dir": args.out_dir,
        "mode": args.mode,
        "min_points": int(args.min_points),
        "datasets_requested": sorted(selected) if selected else None,
        "datasets_processed": dataset_folders,
        "dataset_info_xlsx": dataset_info_path,
        "dataset_info_columns": dataset_info_cols,
        "total_index_rows": int(len(merged)) if len(merged) else 0,
        "ok_samples": int((merged.get("status") == "ok").sum()) if "status" in merged.columns else 0,
        "status_by_dataset_csv": status_csv,
        "failed_reasons_by_dataset_csv": failed_csv,
        "skipped_reasons_by_dataset_csv": skipped_csv,
        "failed_reasons_overall_csv": failed_overall_csv,
        "skipped_reasons_overall_csv": skipped_overall_csv,
    }

    lib.write_json(os.path.join(args.out_dir, "summary.json"), summary)


if __name__ == "__main__":
    main()
