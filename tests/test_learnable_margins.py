from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eyepacs_hybrid_ordinal.losses import (
    CumulativeOrdinalMargins,
    PrototypeContrastiveOrdinalLoss,
    WeightedSupervisedContrastiveOrdinalLoss,
)


def set_margin_values(module: CumulativeOrdinalMargins, values: list[float]) -> None:
    target = torch.tensor(values, dtype=module.raw_margins.dtype)
    shifted = target - module.minimum_margin
    with torch.no_grad():
        module.raw_margins.copy_(torch.log(torch.expm1(shifted)))


def test_margins_stay_above_positive_lower_bound() -> None:
    margins = CumulativeOrdinalMargins(num_classes=5, minimum_margin=0.05)

    with torch.no_grad():
        margins.raw_margins.fill_(-100.0)

    assert torch.all(margins.margin_values() >= 0.05)


def test_distance_accumulates_adjacent_margins() -> None:
    margins = CumulativeOrdinalMargins(num_classes=4, minimum_margin=0.05)
    set_margin_values(margins, [0.2, 0.3, 0.4])

    labels = torch.tensor([0, 1, 2, 3])
    distances = margins(labels, labels)
    expected = torch.tensor(
        [
            [0.0, 0.2, 0.5, 0.9],
            [0.2, 0.0, 0.3, 0.7],
            [0.5, 0.3, 0.0, 0.4],
            [0.9, 0.7, 0.4, 0.0],
        ]
    )

    torch.testing.assert_close(distances, expected)


def test_both_losses_update_the_shared_margins() -> None:
    torch.manual_seed(7)
    margins = CumulativeOrdinalMargins(num_classes=3)
    embeddings = torch.randn(6, 8, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    pcol = PrototypeContrastiveOrdinalLoss(temperature=1.0, num_classes=3)
    scol = WeightedSupervisedContrastiveOrdinalLoss(temperature=1.0, num_classes=3)

    loss = pcol(embeddings, labels, learnable_margins=margins)
    loss = loss + scol(embeddings, labels, learnable_margins=margins)
    loss.backward()

    assert margins.raw_margins.grad is not None
    assert torch.isfinite(margins.raw_margins.grad).all()
    assert torch.count_nonzero(margins.raw_margins.grad) == 2


def test_freeze_disables_margin_gradients() -> None:
    margins = CumulativeOrdinalMargins(num_classes=5)

    margins.freeze()

    assert not margins.raw_margins.requires_grad
