from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eyepacs_hybrid_ordinal.splitting import build_nested_split_indices


def make_grouped_frame() -> pd.DataFrame:
    subject_ids = np.repeat(np.arange(50), 2)
    subject_labels = np.tile(np.arange(5), 10)
    labels = np.repeat(subject_labels, 2)
    return pd.DataFrame({"_subject_id": subject_ids.astype(str), "_label": labels})


def test_nested_split_keeps_train_validation_and_test_subjects_disjoint() -> None:
    frame = make_grouped_frame()

    train_idx, val_idx, test_idx = build_nested_split_indices(
        frame,
        seed=42,
        val_size=0.2,
        num_folds=10,
        fold_index=3,
        subject_independent=True,
    )

    combined = np.concatenate((train_idx, val_idx, test_idx))
    np.testing.assert_array_equal(np.sort(combined), np.arange(len(frame)))
    assert len(np.unique(combined)) == len(frame)

    train_subjects = set(frame.iloc[train_idx]["_subject_id"])
    val_subjects = set(frame.iloc[val_idx]["_subject_id"])
    test_subjects = set(frame.iloc[test_idx]["_subject_id"])
    assert train_subjects.isdisjoint(val_subjects)
    assert train_subjects.isdisjoint(test_subjects)
    assert val_subjects.isdisjoint(test_subjects)


def test_nested_split_is_reproducible() -> None:
    frame = make_grouped_frame()
    kwargs = {
        "seed": 42,
        "val_size": 0.2,
        "num_folds": 10,
        "fold_index": 3,
        "subject_independent": True,
    }

    first = build_nested_split_indices(frame, **kwargs)
    second = build_nested_split_indices(frame, **kwargs)

    for first_indices, second_indices in zip(first, second):
        np.testing.assert_array_equal(first_indices, second_indices)
