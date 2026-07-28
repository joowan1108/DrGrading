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
    num_margins = module.num_classes - 1
    probabilities = (target - module.minimum_margin) / (
        num_margins * (1.0 - module.minimum_margin)
    )
    torch.testing.assert_close(target.sum(), torch.tensor(float(num_margins)))
    assert torch.all(probabilities > 0)
    with torch.no_grad():
        module.raw_margins.copy_(torch.log(probabilities))


def test_margins_initialize_to_unit_ordinal_steps() -> None:
    margins = CumulativeOrdinalMargins(num_classes=5, minimum_margin=0.1)

    torch.testing.assert_close(margins.margin_values(), torch.ones(4))


def test_margins_keep_fixed_sum_and_positive_lower_bound() -> None:
    margins = CumulativeOrdinalMargins(num_classes=5, minimum_margin=0.1)

    with torch.no_grad():
        margins.raw_margins.copy_(torch.tensor([-100.0, 0.0, 1.0, 2.0]))

    values = margins.margin_values()
    assert torch.all(values >= 0.1)
    torch.testing.assert_close(values.sum(), torch.tensor(4.0))


def test_distance_accumulates_adjacent_margins() -> None:
    margins = CumulativeOrdinalMargins(num_classes=4, minimum_margin=0.1)
    set_margin_values(margins, [0.5, 1.0, 1.5])

    labels = torch.tensor([0, 1, 2, 3])
    distances = margins(labels, labels)
    expected = torch.tensor(
        [
            [0.0, 0.5, 1.5, 3.0],
            [0.5, 0.0, 1.0, 2.5],
            [1.5, 1.0, 0.0, 1.5],
            [3.0, 2.5, 1.5, 0.0],
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
