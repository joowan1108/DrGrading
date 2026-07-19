from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def batch_inverse_frequency_sample_weights(
    labels: Sequence[int] | np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Return per-sample inverse-frequency weights normalized within one batch."""
    labels_array = np.asarray(labels, dtype=np.int64)
    if labels_array.ndim != 1:
        raise ValueError("labels must be a one-dimensional sequence.")
    if num_classes < 1:
        raise ValueError("num_classes must be positive.")
    if labels_array.size == 0:
        return np.empty(0, dtype=np.float32)
    if labels_array.min() < 0 or labels_array.max() >= num_classes:
        raise ValueError(f"labels must be in [0, {num_classes - 1}].")

    counts = np.bincount(labels_array, minlength=num_classes).astype(np.float32)
    present = counts > 0
    class_weights = np.zeros(num_classes, dtype=np.float32)
    class_weights[present] = labels_array.size / (present.sum() * counts[present])
    return class_weights[labels_array]
