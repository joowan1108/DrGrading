from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eyepacs_hybrid_ordinal.losses import (
    asymmetric_gaussian_soft_targets,
    asymmetric_soft_label_loss,
)
from eyepacs_hybrid_ordinal.models import AsymmetricGaussianHead


def test_ag_soft_targets_are_normalized_and_peak_at_ground_truth() -> None:
    targets = torch.tensor([2])
    sigma_left = torch.tensor([0.4])
    sigma_right = torch.tensor([1.0])

    distribution = asymmetric_gaussian_soft_targets(
        targets,
        sigma_left,
        sigma_right,
        num_classes=5,
    )

    assert distribution.sum().item() == pytest.approx(1.0)
    assert distribution.argmax(dim=1).item() == 2
    assert distribution[0, 1] < distribution[0, 3]


def test_ag_soft_loss_backpropagates_to_logits_and_dispersions() -> None:
    logits = torch.zeros((2, 5), requires_grad=True)
    sigma_left = torch.tensor([0.5, 0.6], requires_grad=True)
    sigma_right = torch.tensor([1.0, 1.2], requires_grad=True)
    targets = torch.tensor([2, 3])

    loss = asymmetric_soft_label_loss(
        logits,
        targets,
        sigma_left,
        sigma_right,
        soft_target_weight=0.1,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert sigma_left.grad is not None and torch.isfinite(sigma_left.grad).all()
    assert sigma_right.grad is not None and torch.isfinite(sigma_right.grad).all()


def test_ag_soft_mix_interpolates_hard_and_soft_cross_entropy() -> None:
    logits = torch.tensor([[1.5, -0.5, 0.25]], dtype=torch.float32)
    targets = torch.tensor([1])
    sigma_left = torch.tensor([0.5])
    sigma_right = torch.tensor([1.5])

    hard_loss = asymmetric_soft_label_loss(
        logits,
        targets,
        sigma_left,
        sigma_right,
        soft_target_weight=0.0,
    )
    soft_loss = asymmetric_soft_label_loss(
        logits,
        targets,
        sigma_left,
        sigma_right,
        soft_target_weight=1.0,
    )
    mixed_loss = asymmetric_soft_label_loss(
        logits,
        targets,
        sigma_left,
        sigma_right,
        soft_target_weight=0.25,
    )

    assert mixed_loss.item() == pytest.approx(
        0.75 * hard_loss.item() + 0.25 * soft_loss.item()
    )


def test_ag_head_enforces_undergrading_direction_and_bounds() -> None:
    head = AsymmetricGaussianHead(
        input_dim=4,
        hidden_dim=3,
        sigma_min=0.2,
        sigma_max=5.0,
        direction="undergrading",
    )

    sigma_left, sigma_right = head(torch.randn(6, 4))

    assert torch.all(sigma_left <= sigma_right)
    assert torch.all(sigma_left >= 0.2)
    assert torch.all(sigma_right <= 5.0)
