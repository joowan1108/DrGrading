from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eyepacs_hybrid_ordinal.losses import rmse_loss


def test_rmse_loss_penalizes_only_overpredictions() -> None:
    targets = torch.tensor([1.0, 4.0])
    predictions = torch.tensor([4.0, 1.0])

    loss = rmse_loss(predictions, targets, overprediction_weight=2.0, eps=0.0)

    # The +3 error is weighted by 2; the -3 error keeps weight 1.
    assert loss.item() == pytest.approx(((2.0 * 9.0 + 9.0) / 2.0) ** 0.5)


def test_rmse_loss_penalizes_only_underpredictions() -> None:
    targets = torch.tensor([1.0, 4.0])
    predictions = torch.tensor([4.0, 1.0])

    loss = rmse_loss(predictions, targets, underprediction_weight=1.5, eps=0.0)

    # The -3 error is weighted by 1.5; the +3 error keeps weight 1.
    assert loss.item() == pytest.approx(((9.0 + 1.5 * 9.0) / 2.0) ** 0.5)


def test_rmse_loss_weight_one_matches_standard_rmse() -> None:
    targets = torch.tensor([0.0, 2.0, 4.0])
    predictions = torch.tensor([1.0, 2.5, 1.0])

    directional = rmse_loss(predictions, targets, overprediction_weight=1.0, eps=0.0)
    standard = torch.sqrt(torch.mean(torch.square(predictions - targets)))

    assert directional.item() == pytest.approx(standard.item())


def test_rmse_loss_rejects_overprediction_weight_below_one() -> None:
    with pytest.raises(ValueError, match="overprediction_weight"):
        rmse_loss(torch.tensor([1.0]), torch.tensor([0.0]), overprediction_weight=0.5)


def test_rmse_loss_rejects_underprediction_weight_below_one() -> None:
    with pytest.raises(ValueError, match="underprediction_weight"):
        rmse_loss(torch.tensor([0.0]), torch.tensor([1.0]), underprediction_weight=0.5)
