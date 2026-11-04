from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import train_idrid_seen_balanced_softmax as runner
import eyepacs_hybrid_ordinal.balanced_softmax as bs
from eyepacs_hybrid_ordinal.idrid import prepare_idrid
from eyepacs_hybrid_ordinal.utils import load_config
from smoke_test_idrid import write_synthetic_dataset


def source(tmp_path):
    folder = tmp_path / "source"; folder.mkdir()
    cfg = load_config(ROOT / "configs/hybrid_idrid_all_data_no_spatial.yaml")
    (folder / "run_config.json").write_text(json.dumps({"config": cfg, "official_test_used_for_training": True}))
    (folder / "initial_margins.json").write_text(json.dumps({"values": [0.8, 0.7, 0.6, 0.5]}))
    (folder / "final.pt").write_bytes(b"source weights must never be loaded for fresh BS training")
    return folder, cfg


def test_clone_changes_only_bs_and_reuses_actual_initial_margins(tmp_path):
    folder, original = source(tmp_path)
    cfg, saved = runner.configure(folder, tmp_path / "data", tmp_path / "out")
    assert saved == original
    assert cfg["data"]["image_size"] == 224 and cfg["train"]["epochs"] == 500
    assert cfg["loss"]["alpha"] == cfg["loss"]["beta"] == 0.1
    assert cfg["loss"]["mmnp_pair_reduction"] == "mean" and cfg["loss"]["margin_lr"] == 0.001
    assert cfg["loss"]["margin_phase1_min_epochs"] == 50 and cfg["loss"]["margin_phase1_max_epochs"] == 100
    assert cfg["model"]["balanced_softmax"] is True and cfg["loss"]["classification_weight"] == 0.1
    assert cfg["loss"]["margin_init_values"] == [0.8, 0.7, 0.6, 0.5]
    assert "margin_init_min" not in cfg["loss"] and "margin_init_max" not in cfg["loss"]
    assert cfg["provenance"]["source_weights_loaded"] is False
    assert {k:v for k,v in cfg["model"].items() if k != "balanced_softmax"} == original["model"]


class TinyBS(nn.Module):
    def __init__(self, num_classes=5, **kwargs):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 4, 3, stride=8), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.pcol, self.scol = nn.Linear(4, 8), nn.Linear(4, 8)
        self.regression_head = nn.Linear(16, 1)
        self.classification_head = bs.ClassificationHead(16, num_classes)
        nn.init.zeros_(self.regression_head.weight)
        nn.init.constant_(self.regression_head.bias, 2)

    def forward(self, images):
        features = self.encoder(images)
        pcol, scol = self.pcol(features), self.scol(features)
        both = torch.cat([pcol, scol], dim=1)
        return {"pcol": pcol, "scol": scol, "prediction": self.regression_head(both).squeeze(1),
                "logits": self.classification_head(both)}


@pytest.mark.parametrize("duplicate", [False, True])
def test_bs_trains_on_all_rows_then_fresh_inference_and_identical_tsne_policy(tmp_path, monkeypatch, duplicate):
    torch.set_num_threads(2)
    data = tmp_path / "data"
    write_synthetic_dataset(data / "raw")
    if duplicate:
        base = data / "raw/B.%20Disease%20Grading/B. Disease Grading/1. Original Images"
        (base / "b. Testing Set/IDRiD_003.jpg").write_bytes((base / "a. Training Set/IDRiD_001.jpg").read_bytes())
        with pytest.warns(UserWarning):
            prepare_idrid(data / "raw", data / "prepared", strict=False, overlap_policy="report")
    else:
        prepare_idrid(data / "raw", data / "prepared", strict=False)
    folder, cfg = source(tmp_path)
    cfg["data"].update(batch_size=10, num_workers=0, train_batches_per_epoch=2)
    cfg["model"]["pretrained_imagenet"] = False
    cfg["train"].update(epochs=2, amp=False)
    cfg["loss"].update(margin_phase1_min_epochs=1, margin_phase1_max_epochs=1)
    (folder / "run_config.json").write_text(json.dumps({"config": cfg, "official_test_used_for_training": True}))
    constructed = []
    def factory(**kwargs):
        model = TinyBS(**kwargs)
        constructed.append(model.classification_head.fc.weight.detach().clone())
        return model
    monkeypatch.setattr(bs, "BalancedSoftmaxHybridOrdinalNet", factory)
    output = tmp_path / "bs"
    options = dict(device="cpu", num_workers=0, tsne_options={"max_iter": 300}, allow_synthetic=True)
    if duplicate:
        with pytest.warns(UserWarning):
            result = runner.run(folder, data, output, **options)
    else:
        result = runner.run(folder, data, output, **options)
    split = json.loads((output / "training_split.json").read_text())
    assert split["training_samples"] == 30 and len(split["official_test_used_for_training"]) == 10
    assert all(sample in split["train"] for sample in split["official_test_used_for_training"])
    metadata = json.loads((output / "balanced_softmax.json").read_text())
    assert metadata["class_counts"] == [6]*5
    assert metadata["counts_source"] == "actual_training_pool_including_official_test"
    initial = json.loads((output / "initial_margins.json").read_text())
    assert initial["values"] == pytest.approx([0.8, 0.7, 0.6, 0.5])
    checkpoint = torch.load(output / "final.pt", weights_only=True)
    assert checkpoint["optimizer_steps"] == 4 and checkpoint["config"]["loss"]["alpha"] == 0.1
    assert not torch.equal(checkpoint["model"]["classification_head.fc.weight"], constructed[0])
    history = json.loads((output / "metrics_history.json").read_text())["epochs"]
    for row in history:
        loss = row["train"]
        assert loss["total"] == pytest.approx(0.1*loss["pcol"] + 0.1*loss["scol"] + loss["rmse"] + 2*loss["mmnp"] + 0.1*loss["classification"], rel=1e-5)
    report = json.loads((output / "analysis_class1_all/evaluation_metrics.json").read_text())
    assert report["test"]["samples"] == 10 and report["training_performed"] is False
    predictions = pd.read_csv(output / "analysis_class1_all/test_predictions.csv")
    assert predictions.prediction_class.equals(predictions.prediction_continuous.round().clip(0, 4).astype(int))
    coords = pd.read_csv(output / "analysis_class1_all/tsne_coordinates.csv")
    assert (coords.label == 1).sum() == 2
    assert (coords.loc[coords.label != 1, "label"] == coords.loc[coords.label != 1, "prediction_class"]).all()
    assert (output / "analysis_class1_all/tsne_class1_all_other_correct.pdf").is_file()
