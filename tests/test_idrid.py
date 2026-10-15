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
    audit = json.loads((tmp_path / "prepared" / "duplicate_audit.json").read_text())
    assert audit["summary"]["overlapping_test_samples"] == 1
    assert len(audit["cross_split"][0]["train"]) == 1
    assert len(audit["cross_split"][0]["test"]) == 1
    assert not (tmp_path / "prepared" / "manifest.json").exists()


def copy_training_image_to_test(root, *, train_index=1, test_index=1):
    grading = root / "B.%20Disease%20Grading" / "B. Disease Grading" / "1. Original Images"
    source = grading / "a. Training Set" / f"IDRiD_{train_index:03d}.jpg"
    target = grading / "b. Testing Set" / f"IDRiD_{test_index:03d}.jpg"
    target.write_bytes(source.read_bytes())


def test_report_policy_preserves_counts_and_discloses_conflicting_labels(tmp_path):
    write_synthetic_dataset(tmp_path / "raw")
    # Training grade 0 copied onto a test image labelled grade 1.
    copy_training_image_to_test(tmp_path / "raw", test_index=3)
    with pytest.warns(UserWarning, match="1 test images"):
        manifest = prepare_idrid(tmp_path / "raw", tmp_path / "prepared", strict=False, overlap_policy="report")
    with pytest.warns(UserWarning, match="1 test images"):
        frames, _ = train_idrid.load_prepared(tmp_path, allow_synthetic=True)
    assert len(frames["train"]) == 20
    assert len(frames["test"]) == 10
    assert frames["test"]["_overlaps_train"].sum() == 1
    assert frames["test"].iloc[2]["_label"] == 1
    assert manifest["duplicate_summary"]["cross_split_label_conflict_groups"] == 1


def test_missing_preparation_explains_how_to_retry(tmp_path):
    with pytest.raises(FileNotFoundError, match="preparation has not completed") as caught:
        train_idrid.load_prepared(tmp_path)
    assert "scripts/prepare_idrid.py" in str(caught.value)
    assert "--overlap-policy report" in str(caught.value)


def test_prepared_csv_changes_require_preparation_again(tmp_path):
    write_synthetic_dataset(tmp_path / "raw")
    prepare_idrid(tmp_path / "raw", tmp_path / "prepared", strict=False)
    labels = tmp_path / "prepared" / "train.csv"
    labels.write_text(labels.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="train.csv has changed"):
        train_idrid.load_prepared(tmp_path, allow_synthetic=True)


def test_internal_selection_keeps_duplicate_content_groups_together():
    frame = pd.DataFrame({
        "_label": [grade for grade in range(5) for _ in range(10)],
        "sha256": [f"{grade}-{i // 2}" for grade in range(5) for i in range(10)],
    })
    train, val, method = train_idrid.selection_indices(frame, val_size=0.2, seed=42)
    assert method == "sha256_grouped_stratified"
    assert set(frame.iloc[train]["sha256"]).isdisjoint(frame.iloc[val]["sha256"])
    assert set(frame.iloc[train]["_label"]) == set(range(5))
    assert set(frame.iloc[val]["_label"]) == set(range(5))


def test_internal_selection_groups_content_even_when_labels_disagree():
    frame = pd.DataFrame({
        "_label": [grade for grade in range(5) for _ in range(10)],
        "sha256": [f"{grade}-{i // 2}" for grade in range(5) for i in range(10)],
    })
    frame.loc[10, "sha256"] = frame.loc[0, "sha256"]
    train, val, method = train_idrid.selection_indices(frame, val_size=0.2, seed=42)
    assert method == "sha256_grouped_class_coverage"
    assert set(frame.iloc[train]["sha256"]).isdisjoint(frame.iloc[val]["sha256"])
    assert set(frame.iloc[train]["_label"]) == set(range(5))
    assert set(frame.iloc[val]["_label"]) == set(range(5))


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


@pytest.mark.parametrize("overlap_count", [0, 1, 10])
def test_full_pipeline_select_refit_freeze_reload_and_metrics(tmp_path, monkeypatch, overlap_count):
    torch.set_num_threads(2)
    write_synthetic_dataset(tmp_path / "raw")
    for index in range(1, overlap_count + 1):
        copy_training_image_to_test(tmp_path / "raw", train_index=index, test_index=index)
    if overlap_count:
        with pytest.warns(UserWarning):
            prepare_idrid(tmp_path / "raw", tmp_path / "prepared", strict=False, overlap_policy="report")
    else:
        prepare_idrid(tmp_path / "raw", tmp_path / "prepared", strict=False)
    monkeypatch.setattr(train_idrid, "HybridOrdinalNet", TinyHybrid)
    cfg = smoke_config(tmp_path, tmp_path / "results")
    with pytest.raises(ValueError, match="official 413/103"):
        train_idrid.run(cfg)
    if overlap_count:
        with pytest.warns(UserWarning, match="test images also occur in training"):
            payload = train_idrid.run(cfg, allow_synthetic=True)
    else:
        payload = train_idrid.run(cfg, allow_synthetic=True)
    results = tmp_path / "results"
    saved_config = json.loads((results / "run_config.json").read_text())["config"]
    assert saved_config["loss"]["alpha"] == 0.1
    assert saved_config["loss"]["beta"] == 0.1
    records = json.loads((results / "metrics_history.json").read_text())["epochs"]
    selection_records = json.loads((results / "selection_history.json").read_text())["epochs"]
    for record in selection_records + records:
        losses = record["train"]
        expected_total = 0.1 * losses["pcol"] + 0.1 * losses["scol"] + losses["rmse"] + 2.0 * losses["mmnp"]
        assert losses["total"] == pytest.approx(expected_total, rel=1e-5)
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
    assert predictions.overlaps_train.sum() == overlap_count
    assert payload["data_integrity"]["full_test_contains_train_duplicates"] is bool(overlap_count)
    subset = payload["test_without_train_overlap"]
    assert subset["excluded_samples"] == overlap_count
    assert subset["samples"] == 10 - overlap_count
    if overlap_count == 10:
        assert subset["metrics"] is None
        assert subset["status"] == "unavailable_all_test_images_overlap"
    else:
        nonoverlap = predictions.loc[~predictions.overlaps_train]
        assert subset["metrics"]["accuracy"] == pytest.approx((nonoverlap.label == nonoverlap.prediction_class).mean())
        assert subset["metrics"]["mae"] == pytest.approx(abs(nonoverlap.label - nonoverlap.prediction_class).mean())
    with pytest.raises(FileExistsError, match="not empty"):
        train_idrid.run(cfg, allow_synthetic=True)
