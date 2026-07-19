from __future__ import annotations

import numpy as np
import pandas as pd


def build_split_indices(
    frame: pd.DataFrame,
    seed: int,
    val_size: float = 0.2,
    num_folds: int = 0,
    fold_index: int = 0,
    subject_independent: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(frame))
    labels = frame["_label"].to_numpy()
    groups = frame["_subject_id"].to_numpy()

    if num_folds and num_folds > 1:
        fold_index = int(fold_index)
        if not 0 <= fold_index < num_folds:
            raise ValueError(f"fold_index must be in [0, {num_folds - 1}], got {fold_index}.")

        if subject_independent:
            try:
                from sklearn.model_selection import StratifiedGroupKFold

                splitter = StratifiedGroupKFold(n_splits=num_folds, shuffle=True, random_state=seed)
                splits = splitter.split(indices, labels, groups)
            except (ImportError, ValueError):
                from sklearn.model_selection import GroupKFold

                splitter = GroupKFold(n_splits=num_folds)
                splits = splitter.split(indices, labels, groups)
        else:
            from sklearn.model_selection import StratifiedKFold

            splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
            splits = splitter.split(indices, labels)

        for current_fold, (train_idx, val_idx) in enumerate(splits):
            if current_fold == fold_index:
                return np.asarray(train_idx), np.asarray(val_idx)
        raise RuntimeError("Failed to create requested fold.")

    try:
        if subject_independent:
            from sklearn.model_selection import GroupShuffleSplit

            splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
            train_idx, val_idx = next(splitter.split(indices, labels, groups))
        else:
            from sklearn.model_selection import train_test_split

            train_idx, val_idx = train_test_split(
                indices,
                test_size=val_size,
                random_state=seed,
                stratify=labels,
            )
    except (ImportError, ValueError):
        rng = np.random.default_rng(seed)
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        split = int(round(len(shuffled) * (1.0 - val_size)))
        train_idx, val_idx = shuffled[:split], shuffled[split:]

    return np.asarray(train_idx), np.asarray(val_idx)


def build_nested_split_indices(
    frame: pd.DataFrame,
    seed: int,
    val_size: float,
    num_folds: int,
    fold_index: int,
    subject_independent: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create inner train/validation splits while reserving the outer fold for testing."""
    if num_folds < 2:
        raise ValueError("Nested cross-validation requires num_folds >= 2.")
    if not 0.0 < val_size < 1.0:
        raise ValueError("val_size must be between 0 and 1.")

    outer_train_idx, test_idx = build_split_indices(
        frame,
        seed=seed,
        num_folds=num_folds,
        fold_index=fold_index,
        subject_independent=subject_independent,
    )
    outer_train_frame = frame.iloc[outer_train_idx].reset_index(drop=True)
    inner_train_idx, val_idx = build_split_indices(
        outer_train_frame,
        seed=seed + int(fold_index) + 1,
        val_size=val_size,
        num_folds=0,
        subject_independent=subject_independent,
    )
    return outer_train_idx[inner_train_idx], outer_train_idx[val_idx], test_idx
