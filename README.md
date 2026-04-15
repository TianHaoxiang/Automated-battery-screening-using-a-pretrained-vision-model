# battery-soh-dino

Battery state-of-health (SOH) classification project based on structured battery curve representations and a DINO-style vision backbone.

## Overview

This repository contains a lightweight project layout for:

- feature extraction from battery datasets
- AMOTF-style tensor generation workflows
- SOH classification training using a DINOv3-based backbone
- dataset-combination sweep experiments for cross-dataset evaluation

The current repository is organized as a **clean project skeleton** extracted from a larger local research workspace.

## Repository Structure

```text
battery-soh-dino/
├── extract_feature/
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
│   ├── soh_dino_amotf_npz_soc_horizontal.py
│   └── soh_dino_amotf_npz_dataset_cycles_all_classes_soc_horizontal.py
├── data/
├── outputs/
└── .gitignore
```

## Main Components

### `extract_feature/`
Scripts for dataset export and feature preparation.

Typical use cases include:

- exporting Battery Archive cells
- extracting dataset-specific metadata or feature tables
- preparing input CSV files for downstream training

### `lib/battery_archive_feature_lib.py`
Shared utility library for battery archive feature processing.

### `train/soh_dino_amotf_npz_soc_horizontal.py`
Main training entry for the **full-data SOC-horizontal SOH classification pipeline**.

It provides the following commands:

- `build_amotf`
- `train`
- `run_all`

### `train/soh_dino_amotf_npz_dataset_cycles_all_classes_soc_horizontal.py`
Dataset sweep entry for gradually expanding the training set across multiple battery datasets.

It provides:

- `build_amotf`
- `dataset_sweep`

## Important Note

The repository now includes the shared AMOTF training core inside `train/`, so the training entry scripts no longer depend on a Python module located outside the repository.

The main internal shared module is:

```python
soh_dino_amotf_npz_partial_cycles_partial_sweep_soc_horizontal
```

This keeps the CLI entry scripts lightweight while allowing the repository to run as a self-contained project.

## Recommended Environment

A typical working environment would include:

- Python 3.10+
- numpy
- pandas
- scikit-learn
- torch
- torchvision
- transformers
- timm
- Pillow
- tqdm

Depending on your exact training configuration, you may also need:

- peft
- matplotlib
- seaborn

## Typical Workflow

### 1. Prepare labels/features

Generate or organize a CSV containing at least:

- `sample_id`
- `original_path`
- `assigned_class`

### 2. Build AMOTF/AMOTF tensors

Example:

```bash
python train/soh_dino_amotf_npz_soc_horizontal.py build_amotf \
  --labels_csv /path/to/soh_classification_results.csv
```

### 3. Train on full data

Example:

```bash
python train/soh_dino_amotf_npz_soc_horizontal.py train \
  --labels_csv /path/to/soh_classification_results.csv \
  --runs_root /path/to/outputs \
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
  --epochs 50
```

### 4. Run build + train + PNG baseline together

```bash
python train/soh_dino_amotf_npz_soc_horizontal.py run_all \
  --labels_csv /path/to/soh_classification_results.csv \
  --runs_root /path/to/outputs \
  --run_name exp_full_soc_horizontal \
  --finetune_backbone \
  --lr 5e-4 \
  --npz_norm log1p_global \
  --npz_global_max_log 10.0 \
  --use_class_weights \
  --backbone_lr_mult 0.1 \
  --lr_scheduler cosine_warmup \
  --lr_warmup_ratio 0.1 \
  --lr_min 1e-6 \
  --epochs 50
```

### 5. Run dataset sweep

```bash
python train/soh_dino_amotf_npz_dataset_cycles_all_classes_soc_horizontal.py dataset_sweep \
  --labels_csv /path/to/soh_classification_results.csv \
  --runs_root /path/to/outputs \
  --run_name exp_dataset_sweep \
  --finetune_backbone \
  --lr 5e-4 \
  --epochs 50
```

## Data and Outputs

The repository keeps `data/` and `outputs/` directories, but they are ignored by Git by default.

This is intended to avoid pushing:

- raw battery data
- generated NPZ tensors
- model checkpoints
- large experiment outputs

## Current Status

This repository already contains:

- project structure cleanup
- extracted feature-processing scripts
- lightweight training entry points
- GitHub upload setup via SSH

What may still be improved later:

- add a `requirements.txt`
- add example labels CSV
- add reproducible environment instructions
- add experiment result snapshots

## License

No license file has been added yet.
If you plan to make the project public for wider reuse, adding a license is recommended.
