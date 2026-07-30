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
    assert torch.all((target > 0) & (target < 1))
    with torch.no_grad():
        module.raw_margins.copy_(torch.logit(target))


def test_margins_initialize_to_configured_value() -> None:
    margins = CumulativeOrdinalMargins(num_classes=5, initial_margin=0.5)

    torch.testing.assert_close(margins.margin_values(), torch.full((4,), 0.5))


def test_margins_are_independently_bounded_without_fixed_sum() -> None:
    margins = CumulativeOrdinalMargins(num_classes=4, initial_margin=0.5)

    with torch.no_grad():
        margins.raw_margins.copy_(torch.tensor([-10.0, 0.0, 2.0]))

    values = margins.margin_values()
    assert torch.all((values > 0.0) & (values < 1.0))
    assert not torch.isclose(values.sum(), torch.tensor(3.0))


def test_distance_accumulates_adjacent_margins() -> None:
    margins = CumulativeOrdinalMargins(num_classes=4, initial_margin=0.5)
    set_margin_values(margins, [0.2, 0.4, 0.8])

    labels = torch.tensor([0, 1, 2, 3])
    distances = margins(labels, labels)
    expected = torch.tensor(
        [
            [0.0, 0.2, 0.6, 1.4],
            [0.2, 0.0, 0.4, 1.2],
            [0.6, 0.4, 0.0, 0.8],
            [1.4, 1.2, 0.8, 0.0],
        ]
    )

    torch.testing.assert_close(distances, expected)


def test_mean_normalized_distance_preserves_baseline_total_scale() -> None:
    margins = CumulativeOrdinalMargins(num_classes=5, initial_margin=0.2)
    set_margin_values(margins, [0.1, 0.2, 0.3, 0.4])

    labels = torch.tensor([0, 1, 2, 3, 4])
    raw_distances = margins(labels, labels)
    normalized_distances = margins(labels, labels, normalize_mean=True)

    torch.testing.assert_close(raw_distances[0, 4], torch.tensor(1.0))
    torch.testing.assert_close(normalized_distances[0, 4], torch.tensor(4.0))
    torch.testing.assert_close(
        normalized_distances[0],
        torch.tensor([0.0, 0.4, 1.2, 2.4, 4.0]),
    )


def test_losses_can_update_shared_margins_when_not_detached() -> None:
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


def test_mmnp_scol_uses_hinge_and_reports_active_violations() -> None:
    margins = CumulativeOrdinalMargins(num_classes=2, initial_margin=0.5)
    embeddings = torch.ones(4, 2, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1])
    scol = WeightedSupervisedContrastiveOrdinalLoss(
        temperature=1.0,
        num_classes=2,
        objective="mmnp",
    )

    loss = scol(embeddings, labels, learnable_margins=margins)
    loss.backward()

    torch.testing.assert_close(loss.detach(), torch.tensor(0.5))
    assert scol.last_active_violation_rate == 1.0
    assert scol.last_boundary_stats is not None
    torch.testing.assert_close(
        scol.last_boundary_stats["comparisons"],
        torch.tensor([8.0]),
    )
    torch.testing.assert_close(
        scol.last_boundary_stats["active"],
        torch.tensor([8.0]),
    )
    torch.testing.assert_close(
        scol.last_boundary_stats["loss_sum"],
        torch.tensor([4.0]),
    )
    assert margins.raw_margins.grad is not None
    assert margins.raw_margins.grad.item() > 0


def test_mmnp_scol_ignores_pairs_that_satisfy_the_margin() -> None:
    margins = CumulativeOrdinalMargins(num_classes=2, initial_margin=0.5)
    embeddings = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]],
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 1, 1])
    scol = WeightedSupervisedContrastiveOrdinalLoss(
        temperature=1.0,
        num_classes=2,
        objective="mmnp",
    )

    loss = scol(embeddings, labels, learnable_margins=margins)

    torch.testing.assert_close(loss.detach(), torch.tensor(0.0))
    assert scol.last_active_violation_rate == 0.0


def test_pcol_can_use_margins_without_updating_them() -> None:
    torch.manual_seed(11)
    margins = CumulativeOrdinalMargins(num_classes=3, initial_margin=0.2)
    embeddings = torch.randn(6, 8, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    pcol = PrototypeContrastiveOrdinalLoss(temperature=1.0, num_classes=3)

    loss = pcol(
        embeddings,
        labels,
        learnable_margins=margins,
        detach_learnable_margins=True,
    )
    loss.backward()

    assert margins.raw_margins.grad is None
    assert embeddings.grad is not None


def test_mmnp_counts_distant_pairs_at_every_crossed_boundary() -> None:
    margins = CumulativeOrdinalMargins(num_classes=3, initial_margin=0.2)
    embeddings = torch.ones(4, 2, requires_grad=True)
    labels = torch.tensor([0, 0, 2, 2])
    scol = WeightedSupervisedContrastiveOrdinalLoss(
        temperature=1.0,
        num_classes=3,
        objective="mmnp",
    )

    loss = scol(embeddings, labels, learnable_margins=margins)

    torch.testing.assert_close(loss.detach(), torch.tensor(0.4))
    assert scol.last_boundary_stats is not None
    torch.testing.assert_close(
        scol.last_boundary_stats["comparisons"],
        torch.tensor([8.0, 8.0]),
    )
    torch.testing.assert_close(
        scol.last_boundary_stats["active"],
        torch.tensor([8.0, 8.0]),
    )


def test_scol_can_use_margins_without_updating_them() -> None:
    torch.manual_seed(13)
    margins = CumulativeOrdinalMargins(
        num_classes=3,
        initial_margin=0.2,
        minimum_margin=0.05,
    )
    embeddings = torch.randn(6, 8, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    scol = WeightedSupervisedContrastiveOrdinalLoss(
        temperature=1.0,
        num_classes=3,
        objective="logsumexp",
    )

    loss = scol(
        embeddings,
        labels,
        learnable_margins=margins,
        detach_learnable_margins=True,
    )
    loss.backward()

    assert margins.raw_margins.grad is None
    assert embeddings.grad is not None


def test_margin_floor_prevents_complete_collapse() -> None:
    margins = CumulativeOrdinalMargins(
        num_classes=3,
        initial_margin=0.2,
        minimum_margin=0.05,
    )
    with torch.no_grad():
        margins.raw_margins.fill_(-100.0)

    values = margins.margin_values()

    assert torch.all(values >= 0.05)
    assert torch.all(values < 1.0)


def test_margin_initialization_jitter_breaks_symmetry_reproducibly() -> None:
    torch.manual_seed(17)
    first = CumulativeOrdinalMargins(
        num_classes=5,
        initial_margin=0.2,
        minimum_margin=0.05,
        initial_margin_jitter=0.02,
    )
    torch.manual_seed(17)
    second = CumulativeOrdinalMargins(
        num_classes=5,
        initial_margin=0.2,
        minimum_margin=0.05,
        initial_margin_jitter=0.02,
    )

    first_values = first.margin_values()
    second_values = second.margin_values()
    torch.testing.assert_close(first_values, second_values)
    assert torch.all((first_values >= 0.18) & (first_values <= 0.22))
    assert torch.unique(first_values).numel() > 1
