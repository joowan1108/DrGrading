from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eyepacs_hybrid_ordinal.ordinal_metrics import ordinal_error_distribution
from eyepacs_hybrid_ordinal.metrics import aggregate_predictions


def test_ordinal_error_distribution_separates_error_distances() -> None:
    predictions = np.asarray([0, 1, 2, 3, 4, 4], dtype=np.int64)
    targets = np.asarray([0, 0, 4, 1, 3, 4], dtype=np.int64)

    result = ordinal_error_distribution(predictions, targets)

    assert result == {
        "correct_count": 2,
        "adjacent_count": 2,
        "non_adjacent_count": 2,
        "within_one_class_count": 4,
        "correct_rate": 1.0 / 3.0,
        "adjacent_rate": 1.0 / 3.0,
        "non_adjacent_rate": 1.0 / 3.0,
        "within_one_class_rate": 2.0 / 3.0,
    }
    assert (
        result["correct_rate"]
        + result["adjacent_rate"]
        + result["non_adjacent_rate"]
        == 1.0
    )


def test_ordinal_error_distribution_rejects_empty_inputs() -> None:
    with np.testing.assert_raises(ValueError):
        ordinal_error_distribution(np.asarray([]), np.asarray([]))


def test_aggregate_predictions_keeps_global_and_per_class_ordinal_metrics() -> None:
    result = aggregate_predictions(
        predictions=[0.0, 2.0, 2.0, 4.0, 2.0],
        targets=[0, 1, 2, 3, 4],
        num_classes=5,
    )

    assert result["correct_count"] == 2
    assert result["adjacent_count"] == 2
    assert result["non_adjacent_count"] == 1
    assert result["within_one_class_count"] == 4
    assert result["per_class"]["1"]["accuracy"] == 0.0
    assert result["per_class"]["1"]["adjacent_rate"] == 1.0
    assert result["per_class"]["4"]["non_adjacent_rate"] == 1.0
