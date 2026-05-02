# battery-soh-dino

Battery state-of-health (SOH) classification from voltage/current cycling data, using **amplitude-aware multi-span ordinal transition field (AMOTF)** tensors and a **DINOv3-style** pretrained vision backbone. The pipeline turns 1D curves into image-like tensors (including compressed NPZ variants), then trains a classifier with optional backbone fine-tuning, class weighting, and cross-dataset sweep utilities. It is designed to work with metadata and exports aligned with **Battery Archive**–style cell and cycle tables.

## Highlights

* Structured conversion of 1D battery voltage/current curves into AMOTF tensors (PNG / NPZ).
* SOH classification with Hugging Face `transformers` backbones and optional LoRA / full fine-tuning.
* Command-line workflows for AMOTF building, `train` / `run_all`, and progressive **dataset-sweep** experiments.
* Feature-export scripts under `extract_feature/` for preparing labels and paths from multi-source archives.

## 1. Setup

### 1.1 Environments

This project is typically run in the conda environment used for the manuscript experiments:

```bash
conda create -n dinov3 python==3.10.19
conda activate dinov3
```

### 1.2 Python requirements

The repository does not ship a pinned `requirements.txt`; the following versions were recorded from the active **`dinov3`** environment (`python -V` and each package’s `__version__`). Reinstalls may differ by platform and wheel index.

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

Use the scripts in `extract_feature/` to export cells, build feature tables, and produce a **labels CSV** for training. At minimum, downstream training expects columns such as:

* `sample_id`
* `original_path`
* `assigned_class`

## 3. Demo

We provide a lightweight **command-line** workflow for AMOTF generation, SOH classification, and dataset-sweep experiments.

1. **Prepare labels.** Run the appropriate `extract_feature/*.py` helpers (or your own pipeline) so that `soh_classification_results.csv` (or equivalent) lists `sample_id`, `original_path`, and `assigned_class` for the cells you keep.
2. **Build AMOTF tensors** (writes under each sample’s `amotf/` directory, including `amotf_npz`):

   ```bash
   python train/soh_dino_amotf_npz_soc_horizontal.py build_amotf \
     --labels_csv /path/to/soh_classification_results.csv
   ```

3. **Train on the full dataset** (example; adjust paths, `run_name`, and schedule flags as needed):

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

4. **Optional — build and train in one shot** (`run_all` builds AMOTF, trains AMOTF-NPZ and PNG baselines, and writes a small comparison table):

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

5. **Optional — dataset-combination sweep** (progressively adds datasets; see script header for full flags):

   ```bash
   python train/soh_dino_amotf_npz_dataset_cycles_all_classes_soc_horizontal.py dataset_sweep \
     --labels_csv /path/to/soh_classification_results.csv \
     --runs_root /path/to/outputs \
     --run_name exp_dataset_sweep \
     --finetune_backbone \
     --lr 5e-4 \
     --epochs 50
   ```

**Note:** Training pulls pretrained weights from Hugging Face unless you pass `--hf_local_only` after caching models offline. Neural runs are stochastic; small metric differences between machines are normal.

## 4. Repository structure

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
│   ├── soh_dino_amotf_npz_horiz_train_core.py
│   ├── soh_dino_amotf_npz_soc_horizontal.py
│   └── soh_dino_amotf_npz_dataset_cycles_all_classes_soc_horizontal.py
├── data/
├── outputs/
└── .gitignore
```

### Main folders

* `extract_feature/`: dataset export, filtering, and feature-table preparation.
* `lib/`: shared utilities for Battery Archive feature processing.
* `train/`: AMOTF build + DINO training entry points (see **§5**).
* `data/`: expected location for local datasets or intermediate CSV files.
* `outputs/`: expected location for tensors, checkpoints, logs, and summaries (ignored by Git by default).

## 5. Training entry points

* **`soh_dino_amotf_npz_soc_horizontal.py`** — self-contained CLI for `build_amotf`, `train`, and `run_all` on full-cycle AMOTF NPZ / PNG inputs.
* **`soh_dino_amotf_npz_dataset_cycles_all_classes_soc_horizontal.py`** — self-contained CLI for `build_amotf` plus **dataset_sweep** over fixed dataset combinations.
* **`soh_dino_amotf_npz_horiz_train_core.py`** — shared implementation module retained alongside the two public CLIs as the minimal upload set for the training code.

Each `train/*.py` file documents additional **command-line examples** in its module header.

## 6. Outputs

Typical artifacts include:

* AMOTF NPZ tensors and optional PNG previews
* run directories with logs and checkpoints
* `summary.json` and sweep result CSVs

Large `data/` and `outputs/` trees are excluded from Git to avoid committing raw curves or bulky experiment trees.

## 7. Current status

Included today:

* feature-processing and dataset-export scripts
* AMOTF tensor generation and DINO-based SOH training CLIs
* dataset-sweep experiment driver

Possible next steps:

* add a pinned `requirements.txt` or `environment.yml`
* ship a small **example** labels CSV for smoke tests
* snapshot one toy run for regression checks

## 8. Acknowledgement

* Upstream public project: [https://github.com/TianHaoxiang/battery-soh-dino](https://github.com/TianHaoxiang/battery-soh-dino)
* README sectioning (Setup / Datasets / Demo) was aligned with the style of [Prediction of second-life battery degradation trajectory using iMOE](https://github.com/terencetaothucb/Prediction-of-second-life-battery-degradation-trajectory-using-iMOE).
* Battery metadata portal: [BatteryArchive.org](https://www.batteryarchive.org/cycle_list.html?t=0001) (see [battery-lcf](https://github.com/battery-lcf)).

## License

No license file has been added yet.  
If you plan to make the project public for wider reuse, adding a license is recommended.
