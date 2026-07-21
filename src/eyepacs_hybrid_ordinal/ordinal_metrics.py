from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def ordinal_error_distribution(
    class_predictions: Sequence[int] | np.ndarray,
    targets: Sequence[int] | np.ndarray,
) -> dict[str, int | float]:
    """Split ordinal predictions into correct, adjacent, and non-adjacent outcomes."""
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
    signed_errors = prediction_array - target_array
    underdiagnosis_mask = signed_errors < 0
    overdiagnosis_mask = signed_errors > 0
    underdiagnosis_count = int(underdiagnosis_mask.sum())
    overdiagnosis_count = int(overdiagnosis_mask.sum())
    mean_underdiagnosis_distance = (
        float((-signed_errors[underdiagnosis_mask]).mean()) if underdiagnosis_count else 0.0
    )
    mean_overdiagnosis_distance = (
        float(signed_errors[overdiagnosis_mask].mean()) if overdiagnosis_count else 0.0
    )
    return {
        "correct_count": correct_count,
        "adjacent_count": adjacent_count,
        "non_adjacent_count": non_adjacent_count,
        "correct_rate": correct_count / total,
        "adjacent_rate": adjacent_count / total,
        "non_adjacent_rate": non_adjacent_count / total,
        "underdiagnosis_count": underdiagnosis_count,
        "overdiagnosis_count": overdiagnosis_count,
        "underdiagnosis_rate": underdiagnosis_count / total,
        "overdiagnosis_rate": overdiagnosis_count / total,
        "mean_underdiagnosis_distance": mean_underdiagnosis_distance,
        "mean_overdiagnosis_distance": mean_overdiagnosis_distance,
    }
