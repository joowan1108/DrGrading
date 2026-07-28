from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def ordinal_error_distribution(
    class_predictions: Sequence[int] | np.ndarray,
    targets: Sequence[int] | np.ndarray,
) -> dict[str, int | float]:
    """Split ordinal predictions into correct, adjacent, and larger errors."""
    prediction_array = np.asarray(class_predictions, dtype=np.int64)
    target_array = np.asarray(targets, dtype=np.int64)
    if prediction_array.ndim != 1 or target_array.ndim != 1:
        raise ValueError("class_predictions and targets must be one-dimensional.")
    if prediction_array.shape != target_array.shape:
        raise ValueError("class_predictions and targets must have the same shape.")
    if prediction_array.size == 0:
        raise ValueError("class_predictions and targets must not be empty.")

    distances = np.abs(prediction_array - target_array)
    total = int(distances.size)
    correct_count = int((distances == 0).sum())
    adjacent_count = int((distances == 1).sum())
    non_adjacent_count = int((distances > 1).sum())
    within_one_class_count = correct_count + adjacent_count
    return {
        "correct_count": correct_count,
        "adjacent_count": adjacent_count,
        "non_adjacent_count": non_adjacent_count,
        "within_one_class_count": within_one_class_count,
        "correct_rate": correct_count / total,
        "adjacent_rate": adjacent_count / total,
        "non_adjacent_rate": non_adjacent_count / total,
        "within_one_class_rate": within_one_class_count / total,
    }
