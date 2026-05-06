#!/usr/bin/env bash
# Single-fold AMOTF NPZ stability train (50 epochs by default), same hyperparameters
# as run_20260505 hardened manifest; background monitor polls train.log.
#
# Typical usage (after: screen -S dinov3  &&  conda activate dinov3):
#   cd "$(dirname "$0")"   # battery-soh-dino
#   ./stability_fold1_50epochs.sh
#
# Optional: ./stability_fold1_50epochs.sh --epochs 2
#           BATTERY_SOH_TRAIN_PYTHON=/path/to/python ./stability_fold1_50epochs.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
exec python -u run_battery_soh_dino.py stability_fold1 "$@"
