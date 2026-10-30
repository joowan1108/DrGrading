from __future__ import annotations

import json
import math
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
import train_idrid_fixed_iterations as heldout
from eyepacs_hybrid_ordinal.idrid import OFFICIAL_COUNTS, prepare_idrid
from eyepacs_hybrid_ordinal.utils import load_config
from smoke_test_idrid import smoke_config, write_synthetic_dataset


class TinyHybrid(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 4, 3, stride=8), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.pcol, self.scol, self.regression = nn.Linear(4, 8), nn.Linear(4, 8), nn.Linear(16, 1)

    def forward(self, images):
        features = self.encoder(images)
        pcol, scol = self.pcol(features), self.scol(features)
        return {"pcol": pcol, "scol": scol, "prediction": self.regression(torch.cat([pcol, scol], dim=1)).squeeze(1)}


def small_config(tmp_path):
    cfg = smoke_config(tmp_path, tmp_path / "results", ROOT / "configs/hybrid_idrid_fixed_iterations_no_spatial.yaml")
    cfg["split"]["val_size"] = 0.0
    cfg["data"]["train_batches_per_epoch"] = 3  # Same as the previous 30-row combined synthetic pool.
    cfg["train"].update(epochs=3, early_stopping_patience=0)
    return cfg


def test_fixed_iterations_config_preserves_previous_optimization_budget_and_model():
    old = load_config(ROOT / "configs/hybrid_idrid_all_data_no_spatial.yaml")
    new = load_config(ROOT / "configs/hybrid_idrid_fixed_iterations_no_spatial.yaml")
    original_rows = sum(sum(counts) for counts in OFFICIAL_COUNTS.values())
    old_steps = math.ceil(original_rows / old["data"]["batch_size"])
    assert new["data"]["train_batches_per_epoch"] == old_steps == 22
    assert old["train"]["epochs"] == new["train"]["epochs"] == 500
    assert old_steps * new["train"]["epochs"] == 11000
    assert old["model"] == new["model"]
    assert old["loss"] == new["loss"]
    assert new["model"]["spatial_attention"] is False
    assert new["model"]["pooling"] == "gem"


@pytest.mark.parametrize("duplicate", [False, True])
def test_heldout_images_never_enter_training_and_steps_stay_fixed(tmp_path, monkeypatch, capsys, duplicate):
    torch.set_num_threads(2)
    raw = tmp_path / "raw"
    write_synthetic_dataset(raw)
    if duplicate:
        grading = raw / "B.%20Disease%20Grading/B. Disease Grading/1. Original Images"
        source = grading / "a. Training Set/IDRiD_001.jpg"
        target = grading / "b. Testing Set/IDRiD_003.jpg"  # Includes conflicting labels; preserve test label.
        target.write_bytes(source.read_bytes())
        with pytest.warns(UserWarning):
            prepare_idrid(raw, tmp_path / "prepared", strict=False, overlap_policy="report")
    else:
        prepare_idrid(raw, tmp_path / "prepared", strict=False)

    def forbidden(*args, **kwargs):
        pytest.fail("No internal validation or train/validation split should run.")

    monkeypatch.setattr(hybrid, "selection_indices", forbidden)
    monkeypatch.setattr(hybrid, "evaluate", forbidden)
    monkeypatch.setattr(hybrid, "HybridOrdinalNet", TinyHybrid)
    calls, step_calls, scheduler_values = [], [], []
    original_loader, original_components, original_scheduler = hybrid.make_loader, hybrid.build_components, hybrid.build_scheduler

    def record_loader(frame, manifest, cfg, *, training):
        loader = original_loader(frame, manifest, cfg, training=training)
        calls.append((training, set(frame.sample_id), set(frame.sha256), len(loader)))
        if not training:
            assert len(step_calls) == 9  # TEST is not even iterated until all training has finished.
        return loader

    def record_components(cfg, device):
        components = original_components(cfg, device)
        optimizer = components[4]
        optimizer.register_step_post_hook(lambda *args, **kwargs: step_calls.append(1))
        return components

    class RecordingScheduler:
        def __init__(self, optimizer, cfg):
            self.inner = original_scheduler(optimizer, cfg)

        def step(self, value):
            scheduler_values.append(value)
            self.inner.step(value)

        def state_dict(self):
            return self.inner.state_dict()

    monkeypatch.setattr(hybrid, "make_loader", record_loader)
    monkeypatch.setattr(hybrid, "build_components", record_components)
    monkeypatch.setattr(hybrid, "build_scheduler", RecordingScheduler)
    cfg = small_config(tmp_path)
    if duplicate:
        with pytest.warns(UserWarning, match="test images also occur in training"):
            payload = heldout.run(cfg, allow_synthetic=True)
    else:
        payload = heldout.run(cfg, allow_synthetic=True)
    output = tmp_path / "results"
    split = json.loads((output / "training_split.json").read_text())
    assert split["training_samples"] == 20 - int(duplicate)
    assert split["test_samples"] == 10
    assert len(split["excluded_training_test_copies"]) == int(duplicate)
    assert split["validation"] == [] and split["official_test_used_for_training"] == []
    assert set(split["train"]).isdisjoint(split["test"])
    assert all(sample.startswith("train:") for sample in split["train"])
    assert len(calls) == 2 and calls[0][0] is True and calls[1][0] is False
    assert calls[0][2].isdisjoint(calls[1][2])
    assert calls[0][3] == 3  # Not ceil(20/10) or ceil(19/10).
    assert len(step_calls) == 9
    assert payload["optimizer_steps"] == 9 and payload["batches_per_epoch"] == 3
    assert payload["planned_optimizer_steps"] == 9 and payload["final_epoch"] == 3
    assert payload["checkpoint"] == "final.pt"
    assert payload["official_test_used_for_training"] is False
    assert payload["evaluation_is_independent"] is True
    assert not payload["data_integrity"]["full_test_contains_train_duplicates"]
    assert payload["test_without_train_overlap"]["excluded_samples"] == 0
    history = json.loads((output / "metrics_history.json").read_text())["epochs"]
    assert [row["optimizer_steps"] for row in history] == [3, 6, 9]
    assert [row["phase"] for row in history] == ["phase1_joint", "phase2_frozen", "phase2_frozen"]
    assert scheduler_values == [row["train"]["rmse"] for row in history]
    assert not any("val" in row for row in history)
    predictions = pd.read_csv(output / "predictions.csv")
    class_metrics = pd.read_csv(output / "per_class_metrics.csv")
    assert class_metrics["class"].tolist() == list(range(5))
    assert not predictions.overlaps_train.any()
    for grade in range(5):
        subset = predictions.loc[predictions.label == grade]
        correct = int((subset.prediction_class == grade).sum())
        stats = payload["test"]["per_class"][str(grade)]
        assert stats["support"] == len(subset)
        assert stats["correct_count"] == correct
        assert stats["accuracy"] == pytest.approx(correct / len(subset))
        assert stats["accuracy_percent"] == pytest.approx(100 * correct / len(subset))
        assert class_metrics.iloc[grade].accuracy_percent == pytest.approx(stats["accuracy_percent"])
    assert payload["test"]["accuracy"] == pytest.approx((predictions.label == predictions.prediction_class).mean())
    assert np.asarray(payload["test"]["confusion_matrix"]).sum() == 10
    checkpoint = torch.load(output / "final.pt", weights_only=True)
    assert checkpoint["official_test_used_for_training"] is False
    assert checkpoint["optimizer_steps"] == 9
    assert checkpoint["margin_state"]["frozen"] is True
    assert not (output / "best.pt").exists()
    stdout = capsys.readouterr().out
    assert all(f"test Class {grade}: Accuracy=" in stdout for grade in range(5))
    with pytest.raises(FileExistsError, match="not empty"):
        heldout.run(cfg, allow_synthetic=True)


@pytest.mark.parametrize("count", [0, -1, 1.5, True, None])
def test_fixed_iterations_rejects_missing_or_invalid_step_counts(tmp_path, count):
    cfg = small_config(tmp_path)
    cfg["data"]["train_batches_per_epoch"] = count
    with pytest.raises(ValueError, match="train_batches_per_epoch"):
        heldout.run(cfg)


def test_per_class_output_handles_missing_class_without_fabricating_accuracy(tmp_path):
    metrics = hybrid.test_metrics([0.1, 1.2, 0.2], [0, 1, 1])
    hybrid.report_per_class(metrics, tmp_path)
    assert metrics["per_class"]["2"]["accuracy_percent"] is None
    frame = pd.read_csv(tmp_path / "per_class_metrics.csv")
    assert len(frame) == 5
    assert frame.iloc[2].support == 0 and pd.isna(frame.iloc[2].accuracy_percent)
