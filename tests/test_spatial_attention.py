from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eyepacs_hybrid_ordinal.models import HybridOrdinalNet


def test_spatial_attention_efficientnet_forward_shapes() -> None:
    model = HybridOrdinalNet(
        backbone="efficientnet_b0",
        projection_hidden_dim=16,
        projection_dim=8,
        regression_hidden_dim=16,
        regression_input="projection_concat",
        spatial_attention=True,
        pooling="gem",
    ).eval()

    with torch.no_grad():
        outputs = model(torch.randn(2, 3, 64, 64))

    assert outputs["features"].shape == (2, model.feature_dim)
    assert outputs["pcol"].shape == (2, 8)
    assert outputs["scol"].shape == (2, 8)
    assert outputs["prediction"].shape == (2,)
    assert model.global_pool is not None
    assert model.global_pool.p.requires_grad
