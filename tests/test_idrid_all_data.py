from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import train_idrid as hybrid
import train_idrid_all_data as all_data
from eyepacs_hybrid_ordinal.idrid import prepare_idrid
from eyepacs_hybrid_ordinal.models import GeMPooling, HybridOrdinalNet
from eyepacs_hybrid_ordinal.utils import load_config
from smoke_test_idrid import smoke_config, write_synthetic_dataset


class TinyHybrid(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, 3, stride=8), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.pcol, self.scol = nn.Linear(4, 8), nn.Linear(4, 8)
        self.regression = nn.Linear(16, 1)

    def forward(self, images):
        features = self.features(images)
        pcol, scol = self.pcol(features), self.scol(features)
        return {"pcol": pcol, "scol": scol, "prediction": self.regression(torch.cat([pcol, scol], dim=1)).squeeze(1)}


def small_config(tmp_path):
    cfg = smoke_config(tmp_path, tmp_path / "results", ROOT / "configs/hybrid_idrid_all_data_no_spatial.yaml")
    cfg["split"]["val_size"] = 0.0
    return cfg


@pytest.mark.parametrize("name", [
    "hybrid_idrid_efficientnet_v2_s.yaml", "hybrid_idrid_random_margin_long.yaml",
    "hybrid_idrid_all_data_no_spatial.yaml",
])
def test_idrid_configs_keep_efficientnet_gem_but_remove_spatial(name):
    cfg = load_config(ROOT / "configs" / name)
    assert cfg["model"]["backbone"] == "efficientnet_v2_s"
    assert cfg["model"]["pooling"] == "gem"
    assert cfg["model"]["pretrained_imagenet"] is True
    assert cfg["model"]["spatial_attention"] is False
    assert "no_spatial" in cfg["train"]["output_dir"]


def test_real_model_has_gem_without_spatial():
    torch.set_num_threads(2)
    cfg = load_config(ROOT / "configs/hybrid_idrid_all_data_no_spatial.yaml")
    options = {key: value for key, value in cfg["model"].items() if key != "num_classes"}
    options["pretrained_imagenet"] = False  # Offline test; never download weights.
    model = HybridOrdinalNet(**options)
    assert model.spatial_attention is None
    assert isinstance(model.global_pool, GeMPooling)


@pytest.mark.parametrize("duplicate", [False, True])
def test_all_data_training_no_validation_final_checkpoint_and_seen_test(tmp_path, monkeypatch, capsys, duplicate):
    torch.set_num_threads(2)
    raw = tmp_path / "raw"
    write_synthetic_dataset(raw)
    if duplicate:
        grading = raw / "B.%20Disease%20Grading/B. Disease Grading/1. Original Images"
        source = grading / "a. Training Set/IDRiD_001.jpg"
        target = grading / "b. Testing Set/IDRiD_003.jpg"  # Conflicting supplied label.
        target.write_bytes(source.read_bytes())
        with pytest.warns(UserWarning):
            prepare_idrid(raw, tmp_path / "prepared", strict=False, overlap_policy="report")
    else:
        prepare_idrid(raw, tmp_path / "prepared", strict=False)

    def forbidden(*args, **kwargs):
        pytest.fail("Internal validation/splitting must never run in the all-data protocol.")

    monkeypatch.setattr(hybrid, "HybridOrdinalNet", TinyHybrid)
    monkeypatch.setattr(hybrid, "selection_indices", forbidden)
    monkeypatch.setattr(hybrid, "evaluate", forbidden)
    batches = []
    real_loader = hybrid.make_loader

    def record_loader(frame, manifest, cfg, *, training):
        batches.append((training, set(frame["sample_id"])))
        return real_loader(frame, manifest, cfg, training=training)

    monkeypatch.setattr(hybrid, "make_loader", record_loader)
    scheduler_values = []
    real_scheduler = hybrid.build_scheduler

    class RecordingScheduler:
        def __init__(self, optimizer, cfg):
            self.inner = real_scheduler(optimizer, cfg)

        def step(self, value):
            scheduler_values.append(value)
            self.inner.step(value)

        def state_dict(self):
            return self.inner.state_dict()

    monkeypatch.setattr(hybrid, "build_scheduler", RecordingScheduler)
    cfg = small_config(tmp_path)
    cfg["train"].update(epochs=3, early_stopping_patience=0)
    if duplicate:
        with pytest.warns(UserWarning, match="test images also occur in training"):
            payload = all_data.run(cfg, allow_synthetic=True)
    else:
        payload = all_data.run(cfg, allow_synthetic=True)
    output = tmp_path / "results"
    split = json.loads((output / "training_split.json").read_text())
    train_ids = set(split["train"])
    test_ids = set(split["official_test_used_for_training"])
    assert len(train_ids) == 30 and len(test_ids) == 10
    assert test_ids <= train_ids
    assert split["validation"] == split["independent_test"] == []
    assert split["class_counts"] == [6] * 5
    assert split["source_counts"] == {"train": 20, "test": 10}
    assert split["unique_image_hashes"] == 30 - int(duplicate)
    assert batches == [(True, train_ids), (False, test_ids)]
    history = json.loads((output / "metrics_history.json").read_text())["epochs"]
    assert [row["phase"] for row in history] == ["phase1_joint", "phase2_frozen", "phase2_frozen"]
    assert all("val" not in row for row in history)
    assert history[0]["margins"]["values"] == history[-1]["margins"]["values"]
    assert scheduler_values == [row["train"]["rmse"] for row in history]
    for row in history:
        losses = row["train"]
        assert losses["total"] == pytest.approx(
            0.1 * losses["pcol"] + 0.1 * losses["scol"] + losses["rmse"] + 2 * losses["mmnp"], rel=1e-5,
        )
    assert payload["final_epoch"] == 3  # No early stopping, even with patience zero.
    assert payload["checkpoint"] == "final.pt"
    assert payload["evaluation_is_independent"] is False
    assert payload["official_test_used_for_training"] is True
    assert payload["independent_test"] is None
    assert "test" not in payload
    predictions = pd.read_csv(output / "predictions.csv")
    metrics = payload["seen_test"]
    assert metrics["samples"] == len(predictions) == 10
    assert predictions.included_in_training_pool.all()
    assert metrics["accuracy"] == pytest.approx((predictions.label == predictions.prediction_class).mean())
    assert metrics["mae"] == pytest.approx(abs(predictions.label - predictions.prediction_class).mean())
    assert len(metrics["per_class"]) == 5
    assert np.array(metrics["confusion_matrix"]).sum() == 10
    checkpoint = torch.load(output / "final.pt", weights_only=True)
    assert checkpoint["epoch"] == 3
    assert checkpoint["margin_state"]["frozen"] is True
    assert checkpoint["evaluation_is_independent"] is False
    last = torch.load(output / "last.pt", weights_only=True)
    for key, value in checkpoint["model"].items():
        torch.testing.assert_close(value, last["model"][key])
    assert not (output / "best.pt").exists()
    assert not (output / "best_metrics.json").exists()
    saved_cfg = json.loads((output / "run_config.json").read_text())["config"]
    assert saved_cfg["train"]["early_stopping_patience"] is None
    assert saved_cfg["model"]["spatial_attention"] is False
    captured = capsys.readouterr().out
    assert "NOT an independent test result" in captured
    assert "TRAIN-SEEN test subset (NOT independent)" in captured
    assert "[Phase 1 -> Phase 2]" in captured
    with pytest.raises(FileExistsError, match="not empty"):
        all_data.run(cfg, allow_synthetic=True)


def test_all_data_requires_no_validation_and_a_phase2_epoch(tmp_path):
    cfg = small_config(tmp_path)
    cfg["split"]["val_size"] = 0.2
    with pytest.raises(ValueError, match="NO internal validation"):
        all_data.run(cfg)
    cfg["split"]["val_size"] = 0.0
    cfg["train"]["epochs"] = 1
    with pytest.raises(ValueError, match="Phase 2 can run"):
        all_data.run(cfg)
