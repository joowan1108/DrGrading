from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import train_idrid
import eyepacs_hybrid_ordinal.idrid as idrid
from eyepacs_hybrid_ordinal.idrid import OFFICIAL_COUNTS, extract_grading_archive, prepare_idrid
from eyepacs_hybrid_ordinal.transforms import CropBlackBorder, make_idrid_transform
from smoke_test_idrid import smoke_config, write_synthetic_dataset


def test_official_counts_and_repeated_names_are_split_specific(tmp_path):
    source = tmp_path / "raw"
    write_synthetic_dataset(source, OFFICIAL_COUNTS)
    manifest = prepare_idrid(source, tmp_path / "prepared")
    assert manifest["splits"]["train"]["samples"] == 413
    assert manifest["splits"]["test"]["samples"] == 103
    frames, _ = train_idrid.load_prepared(tmp_path)
    train, test = frames["train"], frames["test"]
    # Names restart at IDRiD_001 in the official test folder; they are NOT patient IDs.
    assert Path(train.iloc[0]["image"]).name == Path(test.iloc[0]["image"]).name
    assert train.iloc[0]["image"] != test.iloc[0]["image"]
    assert set(train["level"]) == set(range(5))  # DME in fixture is always zero.
    assert not set(train["sha256"]) & set(test["sha256"])


def test_wrong_counts_fail_before_manifests_are_written(tmp_path):
    write_synthetic_dataset(tmp_path / "raw")
    with pytest.raises(ValueError, match="Unexpected train class counts"):
        prepare_idrid(tmp_path / "raw", tmp_path / "prepared")
    assert not (tmp_path / "prepared" / "train.csv").exists()


def test_duplicate_image_contents_across_splits_rejected(tmp_path):
    write_synthetic_dataset(tmp_path / "raw")
    matches = list((tmp_path / "raw").rglob("IDRiD_001.jpg"))
    matches[1].write_bytes(matches[0].read_bytes())
    with pytest.raises(ValueError, match="Identical image contents"):
        prepare_idrid(tmp_path / "raw", tmp_path / "prepared", strict=False)


def test_extraction_keeps_only_grading_and_blocks_traversal(tmp_path):
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("B. Disease Grading/readme.txt", "grading")
        handle.writestr("A. Segmentation/image.jpg", "not grading")
    extract_grading_archive(archive, tmp_path / "extracted")
    assert (tmp_path / "extracted" / "B. Disease Grading" / "readme.txt").exists()
    assert not (tmp_path / "extracted" / "A. Segmentation").exists()
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("B. Disease Grading/../../escape.txt", "unsafe")
    with pytest.raises(ValueError, match="Unsafe archive path"):
        extract_grading_archive(archive, tmp_path / "extracted")
    assert not (tmp_path / "escape.txt").exists()


def test_download_runs_on_requested_host_path_and_reuses_complete_extract(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(idrid.shutil, "which", lambda name: "/remote/venv/bin/kaggle")

    def fake_download(command, *, check):
        calls.append(command)
        assert check is True
        assert command[:5] == ["/remote/venv/bin/kaggle", "datasets", "download", "-d", idrid.DATASET]
        archive_dir = Path(command[-1])
        with zipfile.ZipFile(archive_dir / "indian-diabetic-retinopathy-image-dataset.zip", "w") as handle:
            handle.writestr("B. Disease Grading/labels.csv", "synthetic placeholder")
            handle.writestr("A. Segmentation/unneeded.txt", "skip")

    monkeypatch.setattr(idrid.subprocess, "run", fake_download)
    destination = tmp_path / "remote data with spaces"
    raw = idrid.download_idrid(destination)
    assert raw == destination / "raw"
    assert (raw / "B. Disease Grading" / "labels.csv").is_file()
    assert not (raw / "A. Segmentation").exists()
    assert idrid.download_idrid(destination) == raw
    assert len(calls) == 1


def test_idrid_augmentation_and_deterministic_evaluation():
    train = make_idrid_transform(training=True)
    evaluation = make_idrid_transform(training=False)
    transforms = train.transforms
    assert any(isinstance(t, T.RandomHorizontalFlip) for t in transforms)
    assert any(isinstance(t, T.RandomVerticalFlip) for t in transforms)
    assert any(isinstance(t, T.RandomRotation) for t in transforms)
    nested = [inner for outer in transforms if isinstance(outer, T.RandomApply) for inner in outer.transforms]
    assert any(isinstance(t, T.GaussianBlur) for t in nested)
    assert any(isinstance(t, T.ColorJitter) for t in nested)
    assert not any(isinstance(t, CropBlackBorder) for t in transforms)
    image = Image.fromarray(np.random.default_rng(9).integers(0, 256, (71, 93, 3), dtype=np.uint8))
    assert train(image).shape == (3, 224, 224)
    torch.testing.assert_close(evaluation(image), evaluation(image), rtol=0, atol=0)
    assert [type(t) for t in evaluation.transforms] == [T.Resize, T.ToTensor, T.Normalize]


class TinyHybrid(nn.Module):
    """Exercise the real training/loss/phase/checkpoint pipeline without expensive CNNs."""
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 4, 3, stride=8), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.pcol = nn.Linear(4, 8)
        self.scol = nn.Linear(4, 8)
        self.regression = nn.Linear(16, 1)

    def forward(self, images):
        features = self.encoder(images)
        pcol, scol = self.pcol(features), self.scol(features)
        return {"pcol": pcol, "scol": scol, "prediction": self.regression(torch.cat([pcol, scol], dim=1)).squeeze(1)}


def test_full_pipeline_select_refit_freeze_reload_and_metrics(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    write_synthetic_dataset(tmp_path / "raw")
    prepare_idrid(tmp_path / "raw", tmp_path / "prepared", strict=False)
    monkeypatch.setattr(train_idrid, "HybridOrdinalNet", TinyHybrid)
    cfg = smoke_config(tmp_path, tmp_path / "results")
    with pytest.raises(ValueError, match="official 413/103"):
        train_idrid.run(cfg)
    payload = train_idrid.run(cfg, allow_synthetic=True)
    results = tmp_path / "results"
    records = json.loads((results / "metrics_history.json").read_text())["epochs"]
    assert records[0]["margins"]["frozen"] is False
    assert records[1]["margins"]["frozen"] is True
    assert records[0]["margins"]["values"] == records[1]["margins"]["values"]
    initial = train_idrid.build_components(cfg, torch.device("cpu"))[1]
    assert records[0]["margins"]["values"] != initial.margin_values().tolist()
    split = json.loads((results / "selection_split.json").read_text())
    assert set(split["train"]).isdisjoint(split["validation"])
    assert all(s.startswith("train:") for s in split["train"] + split["validation"])
    predictions = pd.read_csv(results / "predictions.csv")
    metrics = payload["test"]
    assert metrics["samples"] == 10
    assert metrics["accuracy"] == pytest.approx((predictions.label == predictions.prediction_class).mean())
    assert metrics["mae"] == pytest.approx(abs(predictions.label - predictions.prediction_class).mean())
    assert np.array(metrics["confusion_matrix"]).sum() == 10
    assert len(metrics["per_class"]) == 5
    with pytest.raises(FileExistsError, match="not empty"):
        train_idrid.run(cfg, allow_synthetic=True)
