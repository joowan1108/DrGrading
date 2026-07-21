from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eyepacs_hybrid_ordinal.ordinal_metrics import ordinal_error_distribution


def test_ordinal_error_distribution_separates_correct_adjacent_and_non_adjacent() -> None:
    predictions = np.asarray([0, 1, 2, 3, 4, 4], dtype=np.int64)
    targets = np.asarray([0, 0, 4, 1, 3, 4], dtype=np.int64)

    result = ordinal_error_distribution(predictions, targets)

    assert result == {
        "correct_count": 2,
        "adjacent_count": 2,
        "non_adjacent_count": 2,
        "correct_rate": 1.0 / 3.0,
        "adjacent_rate": 1.0 / 3.0,
        "non_adjacent_rate": 1.0 / 3.0,
        "underdiagnosis_count": 1,
        "overdiagnosis_count": 3,
        "underdiagnosis_rate": 1.0 / 6.0,
        "overdiagnosis_rate": 0.5,
        "mean_underdiagnosis_distance": 2.0,
        "mean_overdiagnosis_distance": 4.0 / 3.0,
    }
    assert result["correct_rate"] + result["adjacent_rate"] + result["non_adjacent_rate"] == 1.0
    assert result["correct_rate"] + result["underdiagnosis_rate"] + result["overdiagnosis_rate"] == 1.0


def test_ordinal_error_distribution_uses_zero_distance_when_direction_is_absent() -> None:
    result = ordinal_error_distribution(
        np.asarray([0, 1, 2], dtype=np.int64),
        np.asarray([0, 1, 2], dtype=np.int64),
    )

    assert result["underdiagnosis_rate"] == 0.0
    assert result["overdiagnosis_rate"] == 0.0
    assert result["mean_underdiagnosis_distance"] == 0.0
    assert result["mean_overdiagnosis_distance"] == 0.0


def test_ordinal_error_distribution_rejects_empty_inputs() -> None:
    with np.testing.assert_raises(ValueError):
        ordinal_error_distribution(np.asarray([]), np.asarray([]))
