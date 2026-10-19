from __future__ import annotations

import importlib.util
import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = load_module("amotf_v2_builder", ROOT / "tools" / "build_amotf_v2_dense.py")
NORMALIZATION = load_module("amotf_v2_normalization", ROOT / "tools" / "amotf_v2_normalization.py")


class AmotfV2Tests(unittest.TestCase):
    def test_cli_builds_charge_and_discharge_dense_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sample_dir = root / "sample_001"
            sample_dir.mkdir()
            points = 150
            charge_soc = np.linspace(0.0, 100.0, points)
            discharge_soc = np.linspace(100.0, 0.0, points)
            pd.DataFrame(
                {
                    "phase": ["charge"] * points + ["discharge"] * points,
                    "soc_pct": np.concatenate([charge_soc, discharge_soc]),
                    "voltage_V": np.concatenate([3.0 + charge_soc / 100.0, 4.1 - discharge_soc / 120.0]),
                    "current_A": np.concatenate([np.full(points, 1.5), np.full(points, -1.2)]),
                }
            ).to_csv(sample_dir / "curve.csv", index=False)
            labels_csv = root / "labels.csv"
            with labels_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["sample_id", "original_path", "assigned_class"])
                writer.writeheader()
                writer.writerow({"sample_id": "sample_001", "original_path": str(sample_dir), "assigned_class": "A"})

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "run_battery_soh_dino.py"),
                    "build_amotf",
                    "--labels_csv", str(labels_csv),
                    "--num_workers", "1",
                ],
                check=True,
                cwd=ROOT,
                env={
                    **os.environ,
                    "BATTERY_SOH_DINO_DATA_ROOT": str(root),
                    "BATTERY_SOH_ENCODING_DIR": "amotf_v2_dense_m5",
                },
            )
            for phase in ("charge", "discharge"):
                with np.load(sample_dir / "amotf_v2_dense_m5" / f"amotf_{phase}.npz") as payload:
                    self.assertEqual(payload["amotf"].shape, (120, 120, 8))

    def test_duplicate_soc_is_averaged_and_descending_direction_is_retained(self) -> None:
        result = BUILDER.resample_by_soc(
            np.asarray([100.0, 90.0, 90.0, 80.0]),
            np.asarray([1.0, 2.0, 4.0, 5.0]),
            3,
        )
        np.testing.assert_allclose(result, np.asarray([1.0, 3.0, 5.0]), atol=1e-7)

    def test_dense_field_shape_and_range(self) -> None:
        series = np.sin(np.linspace(0.0, 8.0, 140, dtype=np.float64))
        encoded = BUILDER.dense_amotf_single(
            series,
            image_size=120,
            m=5,
            tau=1,
            spans=[1, 2, 4, 8],
            transition_mix=0.50,
            signed_mix=0.25,
            emd_mix=0.25,
        )
        self.assertEqual(encoded.shape, (120, 120, 4))
        self.assertTrue(np.isfinite(encoded).all())
        self.assertGreaterEqual(float(encoded.min()), 0.0)
        self.assertLessEqual(float(encoded.max()), 1.0)

    def test_locked_training_normalization_is_channelwise(self) -> None:
        image = np.stack(
            [np.arange(16, dtype=np.float32).reshape(4, 4), np.full((4, 4), 7.0, dtype=np.float32)],
            axis=2,
        )
        normalized = NORMALIZATION.log1p_then_minmax(image)
        self.assertEqual(normalized.shape, image.shape)
        self.assertAlmostEqual(float(normalized[:, :, 0].min()), 0.0)
        self.assertAlmostEqual(float(normalized[:, :, 0].max()), 1.0)
        np.testing.assert_array_equal(normalized[:, :, 1], np.zeros((4, 4), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
