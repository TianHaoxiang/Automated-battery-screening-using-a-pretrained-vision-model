# Automated-battery-screening-using-a-pretrained-vision-model

Battery state-of-health (SOH) classification from voltage/current cycling data, using **amplitude-aware multi-span ordinal transition field (AMOTF)** tensors and a **DINOv3-style** pretrained vision backbone. The pipeline turns 1D curves into image-like tensors (including compressed NPZ variants), then trains a classifier with optional backbone fine-tuning, class weighting, and cross-dataset sweep utilities. It is designed to work with metadata and exports aligned with **Battery Archive**–style cell and cycle tables.

## Highlights

* Structured conversion of 1D battery voltage/current curves into AMOTF tensors (PNG / NPZ).
* SOH classification with Hugging Face backbones and optional LoRA / full fine-tuning.
* Command-line workflows for AMOTF building, `train` / `run_all`, and progressive **dataset-sweep** experiments.
* Feature-export scripts under `extract_feature/` for preparing labels and paths from multi-source archives.

## 1. Setup

### 1.1 Environments

This project now ships both a conda environment file and a pip requirements file.

Recommended (closest to the manuscript environment):

```bash
conda env create -f environment.yml
conda activate dinov3
```

Alternative pip-based install inside an existing Python 3.10 environment:

```bash
python -m pip install -r requirements.txt
```

The manuscript-style environment name remains:

```bash
conda activate dinov3
```

### 1.2 Python requirements

The pinned versions below are mirrored in `environment.yml` and `requirements.txt`, and were recorded from the active **`dinov3`** environment (`python -V` and each package’s `__version__`). Reinstalls may still differ by platform and wheel index.

* python=3.10.19
* numpy=1.26.4
* pandas=2.3.3
* scikit-learn=1.7.2
* torch=2.5.1+cu121
* torchvision=0.20.1+cu121
* transformers=4.57.3
* timm=1.0.24
* Pillow=12.1.1
* tqdm=4.66.5
* matplotlib=3.8.4
* seaborn=0.13.2
* peft=0.18.1

(`torch` / `torchvision` builds above are CUDA 12.1 wheels.)

## 2. Datasets

Public cycling metadata and archives compatible with this workflow are published on **BatteryArchive.org** (see also the [battery-lcf](https://github.com/battery-lcf) ecosystem):

* [Battery Archive — cycling tests (list view)](https://www.batteryarchive.org/cycle_list.html?t=0001)

## 2.1 Public release links

The canonical public code repository for this project is:

* GitHub repository: [https://github.com/TianHaoxiang/battery-soh-dino](https://github.com/TianHaoxiang/battery-soh-dino)

The Zenodo public release record for the code-and-data package is:

* Zenodo record: [https://zenodo.org/records/20054162](https://zenodo.org/records/20054162)
* DOI: `10.5281/zenodo.20054162`

Release files published through that Zenodo record:

* Full project archive: [battery-soh-dino_project_20260506.tar.gz](https://zenodo.org/records/20054162/files/battery-soh-dino_project_20260506.tar.gz?download=1)
* Full dataset archive: [battery-soh-dino_complete_dataset_20260506.tar.gz](https://zenodo.org/records/20054162/files/battery-soh-dino_complete_dataset_20260506.tar.gz?download=1)
* Extracted-feature archive: [battery-soh-dino_extract_feature_outputs_20260506.tar.gz](https://zenodo.org/records/20054162/files/battery-soh-dino_extract_feature_outputs_20260506.tar.gz?download=1)
* Labels / release metadata bundle: [battery-soh-dino_release_materials_20260506.tar.gz](https://zenodo.org/records/20054162/files/battery-soh-dino_release_materials_20260506.tar.gz?download=1)

The repository itself also tracks the lightweight, GitHub-hosted dataset files under `data/`:

* `data/soh_classification_results_portable.csv` — portable final labels table bundled with the repository. It currently contains the 20225-sample ordered subset aligned to `exp_full_finetune_run_all/labels_intersection.csv` and preserves the published metadata columns such as `sample_id`, `cycle_index`, `original_path`, `matched_cell_id`, `nominal_capacity`, `measured_capacity`, `soh`, `assigned_class`, `copy_status`, and `issues`.
* `data/stability_fold1_split_indices.json` — fixed single-fold split definition for the same 20225-row ordering as `data/soh_classification_results_portable.csv`. It stores `train_idx`, `val_idx`, and `test_idx`, and is the default split consumed by `python run_battery_soh_dino.py stability_fold1 --epochs 50`.

Use the scripts in `extract_feature/` to export cells and build intermediate feature/sample tables. Downstream training still expects a final **labels CSV** with columns such as:

* `sample_id`
* `original_path`
* `assigned_class`

This repository now includes a portable final labels file at:

```text
data/soh_classification_results_portable.csv
```

Its `original_path` values are written relative to a data root (for example `no_title_outputs/features/...`) rather than as machine-specific absolute paths like `F:\...`.

At runtime, the public training scripts resolve those relative paths from:

```bash
export BATTERY_SOH_DINO_DATA_ROOT=/path/to/your/project-data-root
```

If the environment variable is not set, the scripts fall back to the built-in path remapping logic used in the original local environment.

## 3. Reproducible workflow from the repository root

All public workflows can now be launched from the repository root with:

```bash
conda activate dinov3
python run_battery_soh_dino.py --help
```

The root launcher is a thin wrapper around the refactored public scripts under `extract_feature/` and `train/`, so you can either call the launcher or call the underlying scripts directly.

For portable reruns on another machine, set the data root once before running any command:

```bash
export BATTERY_SOH_DINO_DATA_ROOT=/path/to/THX-or-equivalent-root
```

If you have your own absolute-path labels file and want to convert it into the portable format used by this repository:

```bash
python run_battery_soh_dino.py make_portable_labels \
  --in_labels_csv /absolute/path/to/soh_classification_results.csv \
  --reference_labels_csv /absolute/path/to/labels_intersection.csv \
  --out_labels_csv data/soh_classification_results_portable.csv
```

### 3.1 Data preparation

`extract_feature/` is responsible for parsing Battery Archive–style source tables and writing intermediate sample tables such as `index_samples.csv`, `summary.json`, and per-dataset summaries.

From the repository root:

```bash
python run_battery_soh_dino.py extract_features
```

This defaults to:

* `root_dir=/mnt/sdb/THX/Battery_THX_HP_P9000/Battery/dataset/Tao/Battery_Archive/Battery_Archive`
* `out_dir=/mnt/sdb/THX/Battery_THX_HP_P9000/Battery/dataset/Tao/Battery_Archive/Battery_Archive/outputs_analysis/features`

To run only one dataset wrapper:

```bash
python run_battery_soh_dino.py extract_dataset --dataset CALCE
```

**Important:** the training scripts do **not** infer `assigned_class` automatically from the extracted tables. Training requires a final `labels_csv` containing at least `sample_id`, `original_path`, and `assigned_class`. The repository now prefers the bundled portable file by default:

```text
data/soh_classification_results_portable.csv
```

The original local file used to generate it was:

```text
/mnt/sdb/THX/Battery_THX_HP_P9000/no_title_outputs/soh_cmaotn_dino_runs/exp_full_finetune_run_all/labels_intersection.csv
```

The bundled fixed split file paired with that labels table is:

```text
data/stability_fold1_split_indices.json
```

### 3.2 Full-data AMOTF build

```bash
python run_battery_soh_dino.py build_amotf \
  --labels_csv data/soh_classification_results_portable.csv
```

### 3.3 Full-data training

```bash
python run_battery_soh_dino.py train \
  --labels_csv data/soh_classification_results_portable.csv \
  --runs_root /mnt/sdb/THX/Battery_THX_HP_P9000/no_title_outputs/soh_amotf_dino_runs \
  --run_name exp_full_soc_horizontal_npz \
  --input_mode amotf_npz \
  --finetune_backbone \
  --lr 5e-4 \
  --npz_norm log1p_global \
  --npz_global_max_log 10.0 \
  --use_class_weights \
  --backbone_lr_mult 0.1 \
  --lr_scheduler cosine_warmup \
  --lr_warmup_ratio 0.1 \
  --lr_min 1e-6 \
  --epochs 50 \
  --num_workers 0
```

### 3.3.1 Single-Fold (Fixed Split) 50-Epoch Training

If you need a deterministic single-fold reproduction (recommended for “must finish 50 epochs” validation),
use the fixed-split launcher:

```bash
conda activate dinov3
python run_battery_soh_dino.py stability_fold1 --epochs 50
```

### 3.4 One-command `run_all`

This is the refactored root-level equivalent of the legacy long command:

```bash
python run_battery_soh_dino.py run_all \
  --labels_csv data/soh_classification_results_portable.csv \
  --runs_root /mnt/sdb/THX/Battery_THX_HP_P9000/no_title_outputs/soh_amotf_dino_runs \
  --run_name exp_full_finetune_run_all \
  --finetune_backbone \
  --lr 5e-4 \
  --npz_norm log1p_global \
  --npz_global_max_log 10.0 \
  --use_class_weights \
  --backbone_lr_mult 0.1 \
  --lr_scheduler cosine_warmup \
  --lr_warmup_ratio 0.1 \
  --lr_min 1e-6 \
  --epochs 50 \
  --num_workers 0
```

The default `num_workers` in the public training CLIs is now `0` for stability-first reproduction. If a smoke test succeeds on your machine, you can increase it later for throughput.

### 3.4.1 Dataset Combination Sweep (All-Classes Cycles)

The sweep runner is:

* `train/soh_dino_amotf_npz_dataset_cycles_all_classes_soc_horizontal.py`

It trains **multiple** runs (one per sweep step). Each run defaults to `--kfold 5`.
For a real `--epoch 50` sweep, expect it to take significantly longer than the single-run training above.

Single-step, single-fold engineering verification (recommended before a full sweep):

```bash
python train/soh_dino_amotf_npz_dataset_cycles_all_classes_soc_horizontal.py dataset_sweep \
  --labels_csv data/soh_classification_results_portable.csv \
  --runs_root /mnt/sdb/THX/Battery_THX_HP_P9000/no_title_outputs/soh_amotf_dino_runs \
  --run_name exp_dataset_sweep_single_fold_step1 \
  --split_indices_json /mnt/sdb/THX/Battery_THX_HP_P9000/no_title_outputs/soh_amotf_dino_runs/run_20260505_011002/amotf_npz/fold_1/split_indices.json \
  --epoch 50 \
  --kfold 5 \
  --max_steps 1 \
  --num_workers 0 \
  --finetune_backbone \
  --use_class_weights \
  --lr 5e-4 --backbone_lr_mult 0.1 \
  --lr_scheduler cosine_warmup --lr_warmup_ratio 0.1 --lr_min 1e-6 \
  --npz_norm log1p_global --npz_global_max_log 10.0 \
  --hf_local_only
```

Full 5-fold sweep (50 epochs per run):

```bash
python train/soh_dino_amotf_npz_dataset_cycles_all_classes_soc_horizontal.py dataset_sweep \
  --labels_csv data/soh_classification_results_portable.csv \
  --runs_root /mnt/sdb/THX/Battery_THX_HP_P9000/no_title_outputs/soh_amotf_dino_runs \
  --run_name exp_dataset_sweep_full_5fold \
  --epoch 50 \
  --kfold 5 \
  --num_workers 0 \
  --finetune_backbone \
  --use_class_weights \
  --lr 5e-4 --backbone_lr_mult 0.1 \
  --lr_scheduler cosine_warmup --lr_warmup_ratio 0.1 --lr_min 1e-6 \
  --npz_norm log1p_global --npz_global_max_log 10.0 \
  --hf_local_only
```

### 3.5 Engineering quick-judge for the README-style 50-epoch path

If you want a fast, engineering-oriented answer to “can the current one-command 50-epoch AMOTF NPZ training path start cleanly on this machine?”, use the fixed-split single-fold launcher with `--quick_judge` first:

```bash
conda activate dinov3
python run_battery_soh_dino.py stability_fold1 --quick_judge
```

This uses the same root-level launcher and the same training hyperparameters as the fixed-split 50-epoch stability run, but exits early after a real probe of the critical path:

* load labels and resolve paths
* map the external `fold_1/split_indices.json`
* construct dataloaders and the DINOv3-based model
* run real AMOTF NPZ loading for the first training batch
* run one real forward / backward / optimizer step on CUDA/CPU
* run limited validation inference
* write and re-read a checkpoint artifact

On success, the launcher prints and validates:

* `.../amotf_npz/train.log`
* `.../amotf_npz/fold_1/quick_judge_report.json`

`quick_judge_report.json` will contain `"status": "pass"` only if that real probe completed successfully.

You can increase probe depth if needed:

```bash
python run_battery_soh_dino.py stability_fold1 \
  --quick_judge \
  --quick_judge_train_batches 2 \
  --quick_judge_eval_batches 2
```

After the quick judge passes, launch the full fixed-split 50-epoch run:

```bash
python run_battery_soh_dino.py stability_fold1 --epochs 50
```

Important scope note:

* a passing `--quick_judge` is a strong fast check that the README-style launcher, paths, environment, first-batch training path, limited evaluation path, and checkpoint writing path are working
* it does **not** prove that the full 50-epoch run will always complete without later filesystem, hardware, or long-run stability issues

### 3.6 Dataset-combination sweep

If the required AMOTF NPZ files are not present yet, build them first for the sweep script:

```bash
python run_battery_soh_dino.py dataset_build_amotf \
  --labels_csv data/soh_classification_results_portable.csv
```

Then run the sweep:

```bash
python run_battery_soh_dino.py dataset_sweep \
  --labels_csv data/soh_classification_results_portable.csv \
  --runs_root /mnt/sdb/THX/Battery_THX_HP_P9000/no_title_outputs/soh_amotf_dino_runs \
  --run_name exp_dataset_sweep \
  --finetune_backbone \
  --lr 5e-4 \
  --epochs 50 \
  --num_workers 0
```

### 3.7 Reproducibility artifacts

The refactored training scripts now write explicit reproducibility artifacts into the run directories:

* `metadata.json` — hyperparameters, paths, and filtered-sample statistics
* `run_manifest.json` — Python executable, package versions, Git commit, command line, and deterministic settings
* `fold_*/split_indices.json` — exact train/val/test splits for each fold
* `summary.json` — aggregated metrics
* `run_all_manifest.json` — top-level manifest for `run_all`
* `dataset_sweep_manifest.json` — top-level manifest for the dataset sweep
* `data/soh_classification_results_portable.csv` — portable final labels file with machine-independent `original_path`

For the closest possible rerun, keep the following fixed:

* the same `labels_csv`
* the same `BATTERY_SOH_DINO_DATA_ROOT` (or an equivalent directory tree under it)
* the same Battery Archive source tree under the referenced `original_path`
* the same conda environment / package versions
* the same `seed`
* the same pretrained Hugging Face weights (use `--hf_local_only` after caching to avoid online changes)

If the root launcher reports that the child process terminated by `SIGSEGV`, the most likely failure mode is a native PyTorch / dataloader crash rather than a Python exception. The default worker count is therefore conservative (`--num_workers 0`) for reproduction-first runs.

## 4. From-zero reproduction checklist

1. Clone the repository and ensure `data/soh_classification_results_portable.csv` is present.
2. Create the environment with `conda env create -f environment.yml`.
3. Set `BATTERY_SOH_DINO_DATA_ROOT` to the directory that contains `no_title_outputs/`, `Battery/`, and the referenced sample folders.
4. Verify the portable labels resolve on your machine:

```bash
python run_battery_soh_dino.py print_defaults
python run_battery_soh_dino.py --dry_run build_amotf --labels_csv data/soh_classification_results_portable.csv
```

5. Cache the Hugging Face model once if you want offline-stable reruns, then add `--hf_local_only` to training commands.
6. Run a smoke test with the default stable worker setting (`--num_workers 0`).
7. Run the full workflow:

```bash
python run_battery_soh_dino.py run_all \
  --labels_csv data/soh_classification_results_portable.csv \
  --runs_root /path/to/runs \
  --run_name exp_full_finetune_run_all \
  --finetune_backbone \
  --lr 5e-4 \
  --npz_norm log1p_global \
  --npz_global_max_log 10.0 \
  --use_class_weights \
  --backbone_lr_mult 0.1 \
  --lr_scheduler cosine_warmup \
  --lr_warmup_ratio 0.1 \
  --lr_min 1e-6 \
  --epochs 50 \
  --num_workers 0
```

8. Confirm that the run directory contains `run_all_manifest.json`, `metadata.json`, `run_manifest.json`, `fold_*/split_indices.json`, and final `summary.json` files.

## 5. Repository structure

```text
battery-soh-dino/
├── run_battery_soh_dino.py
├── extract_feature/
│   ├── _bootstrap.py
│   ├── export_batteryarchive_all_cells.py
│   ├── extract_Battery_Archive_feature.py
│   ├── extract_CALCE_feature.py
│   ├── extract_HNEI_feature.py
│   ├── extract_Oxford_feature.py
│   ├── extract_SNL_LFP_feature.py
│   ├── extract_SNL_NCA_feature.py
│   ├── extract_SNL_NMC_feature.py
│   ├── extract_UL_Purdue_feature.py
│   └── extract_fm_lists.py
├── lib/
│   └── battery_archive_feature_lib.py
├── train/
│   ├── soh_dino_amotf_npz_horiz_train_core.py
│   ├── soh_dino_amotf_npz_soc_horizontal.py
│   └── soh_dino_amotf_npz_dataset_cycles_all_classes_soc_horizontal.py
├── data/
├── outputs/
└── .gitignore
```

### Main folders

* `run_battery_soh_dino.py`: root-level one-command launcher for public extract/train workflows.
* `extract_feature/`: dataset export, filtering, and feature-table preparation.
* `lib/`: shared utilities for Battery Archive feature processing.
* `train/`: AMOTF build + DINO training entry points (see **§5**).
* `data/`: expected location for local datasets or intermediate CSV files.
* `outputs/`: expected location for tensors, checkpoints, logs, and summaries (ignored by Git by default).

## 6. Training entry points

* **`soh_dino_amotf_npz_soc_horizontal.py`** — self-contained CLI for `build_amotf`, `train`, and `run_all` on full-cycle AMOTF NPZ / PNG inputs.
* **`soh_dino_amotf_npz_dataset_cycles_all_classes_soc_horizontal.py`** — self-contained CLI for `build_amotf` plus **dataset_sweep** over fixed dataset combinations.
* **`soh_dino_amotf_npz_horiz_train_core.py`** — shared implementation module retained alongside the two public CLIs as the minimal upload set for the training code.

Each `train/*.py` file documents additional **command-line examples** in its module header.

## 7. Outputs

Typical artifacts include:

* AMOTF NPZ tensors and optional PNG previews
* run directories with logs and checkpoints
* `summary.json` and sweep result CSVs

Large `data/` and `outputs/` trees are excluded from Git to avoid committing raw curves or bulky experiment trees.

## 8. Current status

Included today:

* feature-processing and dataset-export scripts
* AMOTF tensor generation and DINO-based SOH training CLIs
* dataset-sweep experiment driver

## 9. Acknowledgement

* Upstream public project: [https://github.com/TianHaoxiang/battery-soh-dino](https://github.com/TianHaoxiang/battery-soh-dino)

* Battery metadata portal: [BatteryArchive.org](https://www.batteryarchive.org/cycle_list.html?t=0001) (see [battery-lcf](https://github.com/battery-lcf)).

## License

No license file has been added yet.  

