from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eyepacs_hybrid_ordinal.weighting import batch_inverse_frequency_sample_weights


def test_batch_inverse_frequency_weights_balance_present_classes() -> None:
    labels = np.asarray([0, 0, 0, 0, 1, 1, 2, 2], dtype=np.int64)

    weights = batch_inverse_frequency_sample_weights(labels, num_classes=5)

    np.testing.assert_allclose(weights[labels == 0], 2.0 / 3.0)
    np.testing.assert_allclose(weights[labels == 1], 4.0 / 3.0)
    np.testing.assert_allclose(weights[labels == 2], 4.0 / 3.0)
    np.testing.assert_allclose(weights.mean(), 1.0)


def test_batch_inverse_frequency_weights_are_one_for_balanced_batch() -> None:
    labels = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)

    weights = batch_inverse_frequency_sample_weights(labels, num_classes=3)

    np.testing.assert_array_equal(weights, np.ones_like(weights))
