from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import train_idrid as hybrid
import train_idrid_balanced_softmax as bs_runner
import eyepacs_hybrid_ordinal.balanced_softmax as bs
import eyepacs_hybrid_ordinal.models as models
from eyepacs_hybrid_ordinal.idrid import prepare_idrid
from eyepacs_hybrid_ordinal.utils import load_config
from smoke_test_idrid import smoke_config, write_synthetic_dataset


class TinyBS(nn.Module):
    def __init__(self, num_classes=5, **kwargs):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 4, 3, stride=8), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.pcol, self.scol = nn.Linear(4, 8), nn.Linear(4, 8)
        self.regression_head = nn.Linear(16, 1)
        self.classification_head = bs.ClassificationHead(16, num_classes)

    def forward(self, images):
        features = self.encoder(images)
        pcol, scol = self.pcol(features), self.scol(features)
        shared = torch.cat([pcol, scol], dim=1)
        return {"pcol": pcol, "scol": scol, "prediction": self.regression_head(shared).squeeze(1), "logits": self.classification_head(shared)}


def test_bs_formula_and_gradients_match_teammate_implementation():
    torch.manual_seed(6)
    logits = torch.randn(10, 5, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    counts = [134, 20, 136, 74, 49]
    actual = bs.BalancedSoftmaxLoss(counts)(logits, labels)
    expected = F.cross_entropy(logits + torch.tensor(counts, dtype=torch.float32).clamp_min(1).log(), labels)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, logits, retain_graph=True)[0], torch.autograd.grad(expected, logits)[0],
    )


@pytest.mark.parametrize("regression_input", ["backbone", "projection_concat"])
def test_bs_head_uses_same_features_as_regression_and_updates_shared_layers(monkeypatch, regression_input):
    torch.set_num_threads(2)

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(nn.Conv2d(3, 8, 1), nn.ReLU())
            self.avgpool = nn.AdaptiveAvgPool2d(1)

    monkeypatch.setattr(models, "build_encoder", lambda *args: (Encoder(), 8))
    model = bs.BalancedSoftmaxHybridOrdinalNet(
        backbone="efficientnet_v2_s", pooling="gem", regression_input=regression_input,
        projection_hidden_dim=8, projection_dim=4, regression_hidden_dim=8,
    )
    outputs = model(torch.randn(10, 3, 16, 16))
    assert outputs["logits"].shape == (10, 5)
    assert outputs["prediction"].shape == (10,)
    assert model.classification_head.fc.in_features == 8
    assert model.spatial_attention is None
    loss = bs.BalancedSoftmaxLoss([134, 20, 136, 74, 49])(outputs["logits"], torch.arange(10) % 5)
    loss.backward()
    assert model.classification_head.fc.weight.grad.abs().sum() > 0
    assert model.encoder.features[0].weight.grad.abs().sum() > 0
    assert model.regression_head.net[0].weight.grad is None  # Auxiliary loss does not bypass shared features.
    if regression_input == "projection_concat":
        assert model.pcol_head.net[0].weight.grad.abs().sum() > 0
        assert model.scol_head.net[0].weight.grad.abs().sum() > 0


def test_config_keeps_branch_loss_weights_and_current_idrid_protocol():
    cfg = load_config(ROOT / "configs/hybrid_idrid_balanced_softmax.yaml")
    control = load_config(ROOT / "configs/hybrid_idrid_fixed_iterations_no_spatial.yaml")
    assert cfg["data"] == control["data"] and cfg["train"]["epochs"] == 500
    assert cfg["data"]["train_batches_per_epoch"] == 22
    assert cfg["loss"]["alpha"] == 0.5 and cfg["loss"]["beta"] == 0.1
    assert cfg["loss"]["classification_weight"] == 0.1
    assert cfg["model"]["balanced_softmax"] is True
    assert cfg["model"]["spatial_attention"] is False and cfg["model"]["pooling"] == "gem"
    assert cfg["provenance"]["source_commit"] == "03fd7e064acbfb1e0062d009e6e413ccc58a4a41"


@pytest.mark.parametrize("duplicate,weight", [(False, 0.1), (True, 0.9)])
def test_bs_training_uses_only_clean_train_counts_and_reports_regression_metrics(tmp_path, monkeypatch, capsys, duplicate, weight):
    torch.set_num_threads(2)
    raw = tmp_path / "raw"
    write_synthetic_dataset(raw)
    if duplicate:
        grading = raw / "B.%20Disease%20Grading/B. Disease Grading/1. Original Images"
        (grading / "b. Testing Set/IDRiD_003.jpg").write_bytes((grading / "a. Training Set/IDRiD_001.jpg").read_bytes())
        with pytest.warns(UserWarning):
            prepare_idrid(raw, tmp_path / "prepared", strict=False, overlap_policy="report")
    else:
        prepare_idrid(raw, tmp_path / "prepared", strict=False)
    initial_weights, created = [], []

    def factory(**kwargs):
        model = TinyBS(**kwargs)
        initial_weights.append(model.classification_head.fc.weight.detach().clone())
        created.append(model)
        return model

    monkeypatch.setattr(bs, "BalancedSoftmaxHybridOrdinalNet", factory)

    def forbidden(*args, **kwargs):
        pytest.fail("No internal validation or test-based checkpoint selection is allowed.")

    monkeypatch.setattr(hybrid, "evaluate", forbidden)
    monkeypatch.setattr(hybrid, "selection_indices", forbidden)
    cfg = smoke_config(tmp_path, tmp_path / "results", ROOT / "configs/hybrid_idrid_balanced_softmax.yaml")
    cfg["split"]["val_size"] = 0.0
    cfg["data"]["train_batches_per_epoch"] = 3
    cfg["loss"]["classification_weight"] = weight
    if duplicate:
        with pytest.warns(UserWarning, match="test images also occur"):
            payload = bs_runner.run(cfg, allow_synthetic=True)
    else:
        payload = bs_runner.run(cfg, allow_synthetic=True)
    output = tmp_path / "results"
    metadata = json.loads((output / "balanced_softmax.json").read_text())
    expected_counts = [3 if duplicate else 4, 4, 4, 4, 4]
    assert metadata["class_counts"] == expected_counts  # Never [6,6,6,6,6] from train+test.
    assert payload["balanced_softmax"]["class_counts"] == expected_counts
    assert payload["balanced_softmax"]["classification_weight"] == weight
    assert payload["optimizer_steps"] == 6
    assert payload["official_test_used_for_training"] is False
    assert payload["test"]["samples"] == 10
    assert not torch.equal(initial_weights[0], created[0].classification_head.fc.weight)
    history = json.loads((output / "metrics_history.json").read_text())["epochs"]
    assert [row["phase"] for row in history] == ["phase1_joint", "phase2_frozen"]
    for row in history:
        losses = row["train"]
        assert losses["classification"] > 0
        assert losses["total"] == pytest.approx(
            0.5 * losses["pcol"] + 0.1 * losses["scol"] + losses["rmse"]
            + 2 * losses["mmnp"] + weight * losses["classification"], rel=1e-5,
        )
    predictions = pd.read_csv(output / "predictions.csv")
    expected_grade = predictions.prediction_continuous.round().clip(0, 4).astype(int)
    assert expected_grade.equals(predictions.prediction_class)
    assert payload["test"]["accuracy"] == pytest.approx((predictions.label == expected_grade).mean())
    assert len(pd.read_csv(output / "per_class_metrics.csv")) == 5
    checkpoint = torch.load(output / "final.pt", weights_only=True)
    assert "classification_head.fc.weight" in checkpoint["model"]
    assert checkpoint["balanced_softmax"]["class_counts"] == expected_counts
    assert "classification=" in capsys.readouterr().out


def test_bs_fails_instead_of_silently_ignoring_auxiliary_loss():
    cfg = load_config(ROOT / "configs/hybrid_idrid_balanced_softmax.yaml")
    with pytest.raises(ValueError, match="actual training pool"):
        hybrid.build_components(cfg, torch.device("cpu"))
    cfg["model"]["balanced_softmax"] = False
    with pytest.raises(ValueError, match="model.balanced_softmax"):
        hybrid.build_components(cfg, torch.device("cpu"))
    with pytest.raises(ValueError, match="model.balanced_softmax"):
        bs_runner.run(cfg)
