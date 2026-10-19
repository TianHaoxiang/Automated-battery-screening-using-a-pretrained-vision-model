"""Locked training normalization for amotf_v2_dense_m5 tensors.

Input arrays use H x W x C layout and are expected to be non-negative.
The transform is applied independently to every channel of every image.
"""

from __future__ import annotations

import numpy as np


def log1p_then_minmax(image_hwc: np.ndarray) -> np.ndarray:
    """Apply log1p followed by per-channel min-max normalization."""
    image = np.asarray(image_hwc, dtype=np.float32)
    image = np.log1p(np.maximum(image, 0.0))
    channel_min = np.min(image, axis=(0, 1), keepdims=True)
    channel_max = np.max(image, axis=(0, 1), keepdims=True)
    scale = np.maximum(channel_max - channel_min, 1e-8)
    return np.clip((image - channel_min) / scale, 0.0, 1.0).astype(
        np.float32, copy=False
    )
