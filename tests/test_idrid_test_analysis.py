from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from sklearn.manifold import TSNE  # Load its PCA dependency before the temporary PCA spy.
from sklearn.metrics import cohen_kappa_score
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import train_idrid as hybrid
import evaluate_idrid_test as posthoc
from eyepacs_hybrid_ordinal.idrid import prepare_idrid
from eyepacs_hybrid_ordinal.idrid_analysis import save_correct_test_tsne
from eyepacs_hybrid_ordinal.plot_colors import DR_CLASS_COLORS
from eyepacs_hybrid_ordinal.utils import save_json
from report_idrid_test import report
from smoke_test_idrid import smoke_config, write_synthetic_dataset


TARGETS = np.repeat(np.arange(5), 2)
PREDICTIONS = np.array([0, 1, 1, 0, 2, 4, 3, 1, 4, 2], dtype=float)


def arrays():
    return (np.random.default_rng(12).normal(size=(10, 8)).astype(np.float32), PREDICTIONS,
            TARGETS, [f"testing/image_{i}.jpg" for i in range(10)], [f"test:image_{i}" for i in range(10)])


def test_incorrect_samples_are_removed_before_pca_and_tsne(tmp_path, monkeypatch):
    features, predictions, targets, images, ids = arrays()
    features[1::2] = 1e6  # Wrong predictions must not influence dimensionality reduction.
    expected = features[::2].copy()
    calls = []

    class PCA:
        def __init__(self, **kwargs):pass
        def fit_transform(self, data):
            np.testing.assert_array_equal(data, expected)
            calls.append("pca")
            return data[:, :2]

    class TSNE:
        def __init__(self, **kwargs):
            assert kwargs["random_state"] == 42 and kwargs["perplexity"] < 5
        def fit_transform(self, data):
            np.testing.assert_array_equal(data, expected[:, :2])
            calls.append("tsne")
            return np.arange(10).reshape(5, 2).astype(float)

    monkeypatch.setattr("sklearn.decomposition.PCA", PCA)
    monkeypatch.setattr("sklearn.manifold.TSNE", TSNE)
    metadata = save_correct_test_tsne(tmp_path, features, predictions, targets, images, ids)
    assert calls == ["pca", "tsne"]
    assert metadata["correct_samples"] == 5 and metadata["excluded_incorrect_samples"] == 5
    assert metadata["class_colors"] == {str(k): color for k, color in enumerate(DR_CLASS_COLORS)}
    assert metadata["correct_class_counts"] == {str(k): 1 for k in range(5)}
    frame = pd.read_csv(tmp_path / metadata["coordinates_file"])
    assert frame.sample_id.tolist() == ids[::2]
    assert (frame.label == frame.prediction_class).all()
    with np.load(tmp_path / metadata["embeddings_file"], allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["embeddings"], expected)
        assert archive["sample_ids"].tolist() == ids[::2]
    assert (tmp_path / metadata["pdf_file"]).read_bytes().startswith(b"%PDF")
    with Image.open(tmp_path / metadata["image_file"]) as image:
        assert image.width > 1500 and image.height > 1200


def test_actual_tsne_is_finite_and_reproducible_on_correct_subset(tmp_path):
    first = save_correct_test_tsne(tmp_path / "first", *arrays(), seed=42)
    second = save_correct_test_tsne(tmp_path / "second", *arrays(), seed=42)
    assert first["status"] == second["status"] == "created"
    a = pd.read_csv(tmp_path / "first" / first["coordinates_file"])
    b = pd.read_csv(tmp_path / "second" / second["coordinates_file"])
    assert len(a) == 5 and np.isfinite(a[["tsne_x", "tsne_y"]]).all().all()
    np.testing.assert_allclose(a[["tsne_x", "tsne_y"]], b[["tsne_x", "tsne_y"]])


@pytest.mark.parametrize("correct_count", [0, 1, 2])
def test_too_few_correct_predictions_never_produce_fabricated_tsne(tmp_path, correct_count):
    features, _, targets, images, ids = arrays()
    predictions = (targets + 1) % 5
    predictions[:correct_count] = targets[:correct_count]
    result = save_correct_test_tsne(tmp_path, features, predictions, targets, images, ids)
    assert result["status"] == "skipped_insufficient_correct_samples"
    assert result["correct_samples"] == correct_count
    frame = pd.read_csv(tmp_path / result["coordinates_file"])
    assert len(frame) == correct_count and frame[["tsne_x", "tsne_y"]].isna().all().all()


def test_identical_correct_features_have_an_explicit_skip_reason(tmp_path):
    features, predictions, targets, images, ids = arrays()
    result = save_correct_test_tsne(tmp_path, np.zeros_like(features), predictions, targets, images, ids)
    assert result["status"] == "skipped_identical_embeddings"
    assert result["correct_samples"] == 5


class LookupModel(nn.Module):
    def forward(self, images):
        n = len(images)
        features = torch.arange(n * 8, dtype=torch.float32).reshape(n, 8) + 1
        return {"prediction": torch.tensor(PREDICTIONS[:n], dtype=torch.float32),
                "pcol": features[:, :4], "scol": features[:, 4:]}


def prepared(tmp_path):
    write_synthetic_dataset(tmp_path / "data" / "raw")
    prepare_idrid(tmp_path / "data" / "raw", tmp_path / "data" / "prepared", strict=False)
    return hybrid.load_prepared(tmp_path / "data", allow_synthetic=True)


def test_metrics_use_all_test_images_while_only_correct_points_are_plotted(tmp_path, monkeypatch):
    frames, manifest = prepared(tmp_path)
    test = frames["test"]
    monkeypatch.setattr(hybrid, "make_loader", lambda *args, **kwargs: [(
        torch.zeros(10, 3, 8, 8), torch.tensor(TARGETS), test._image_id.tolist(),
    )])
    cfg = smoke_config(tmp_path / "data", tmp_path / "results")
    payload = hybrid.evaluate_test(LookupModel(), test, manifest, cfg, tmp_path, torch.device("cpu"),
                                   False, training_frame=frames["train"], require_heldout=True)
    metrics = payload["test"]
    assert metrics["samples"] == 10 and metrics["accuracy"] == 0.5
    assert metrics["quadratic_weighted_kappa"] == pytest.approx(
        cohen_kappa_score(TARGETS, PREDICTIONS.astype(int), weights="quadratic", labels=list(range(5)))
    )
    assert payload["tsne_correct_test"]["correct_samples"] == 5
    overall = pd.read_csv(tmp_path / "test_metrics.csv").iloc[0]
    assert overall.samples == 10 and overall.accuracy == 0.5
    assert overall.qwk == pytest.approx(metrics["quadratic_weighted_kappa"])
    classes = pd.read_csv(tmp_path / "per_class_metrics.csv")
    assert classes.support.tolist() == [2] * 5
    assert classes.accuracy.tolist() == [0.5] * 5
    assert classes.iloc[3].precision == 1.0 and classes.iloc[3].recall == 0.5
    assert classes.iloc[3].f1 == pytest.approx(2 / 3)


def test_existing_final_checkpoint_can_be_analyzed_without_training_or_prepare(tmp_path, monkeypatch):
    frames, manifest = prepared(tmp_path)
    run_dir = tmp_path / "original_run";run_dir.mkdir()
    cfg = smoke_config(tmp_path / "data", run_dir)
    checkpoint = {"model": LookupModel().state_dict(), "config": cfg, "epoch": 2,
                  "official_test_used_for_training": False,
                  "protocol": "official_train_only_no_validation_fixed_iterations_final_epoch_heldout_test"}
    torch.save(checkpoint, run_dir / "final.pt")
    save_json(run_dir / "run_config.json", {"config": cfg, "git_commit": "saved-training-commit"})
    save_json(run_dir / "training_split.json", {
        "train": frames["train"].sample_id.tolist(), "test": frames["test"].sample_id.tolist(),
    })
    original = {p.name: p.read_bytes() for p in run_dir.iterdir()}
    monkeypatch.setattr(posthoc, "build_evaluation_model", lambda *args: LookupModel())
    monkeypatch.setattr(hybrid, "make_loader", lambda *args, **kwargs: [(
        torch.zeros(10, 3, 8, 8), torch.tensor(TARGETS), frames["test"]._image_id.tolist(),
    )])
    def forbidden(*args, **kwargs):pytest.fail("Analysis must not train, select a new model or prepare data.")
    monkeypatch.setattr(hybrid, "train_one_epoch", forbidden)
    monkeypatch.setattr("eyepacs_hybrid_ordinal.idrid.prepare_idrid", forbidden)
    result = posthoc.run(run_dir, output_dir=tmp_path / "analysis", device="cpu", allow_synthetic=True)
    assert result["training_performed"] is False and result["test"]["samples"] == 10
    assert result["tsne_correct_test"]["correct_samples"] == 5
    assert {p.name: p.read_bytes() for p in run_dir.iterdir()} == original
    assert report(tmp_path / "analysis")["test"]["accuracy"] == 0.5


def test_seen_test_checkpoint_cannot_be_relabelled_as_heldout_analysis(tmp_path):
    torch.save({"official_test_used_for_training": True}, tmp_path / "final.pt")
    with pytest.raises(ValueError, match="TRAIN-SEEN checkpoint"):
        posthoc.run(tmp_path)


def test_wrong_run_folder_has_an_actionable_error_before_loading_torch(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):pytest.fail("Do not load a checkpoint for a nonexistent run folder.")
    monkeypatch.setattr(posthoc.torch, "load", forbidden)
    with pytest.raises(FileNotFoundError, match="ACTUAL training output folder") as caught:
        posthoc.run(tmp_path / "example_name")
    assert "--list-runs" in str(caught.value)


def test_incomplete_training_is_explained_without_substituting_last_checkpoint(tmp_path, monkeypatch):
    (tmp_path / "last.pt").write_bytes(b"not a model; must never be loaded")
    save_json(tmp_path / "run_config.json", {"config": {"train": {"epochs": 500}}})
    save_json(tmp_path / "metrics.json", {"epoch": 120})
    def forbidden(*args, **kwargs):pytest.fail("An intermediate last.pt must not substitute for final.pt.")
    monkeypatch.setattr(posthoc.torch, "load", forbidden)
    with pytest.raises(FileNotFoundError, match="120/500") as caught:
        posthoc.run(tmp_path)
    assert "unfinished or interrupted" in str(caught.value)
    assert "do not rename" in str(caught.value)


def test_best_checkpoint_is_not_silently_treated_as_final(tmp_path):
    (tmp_path / "best.pt").write_bytes(b"not the final-epoch protocol")
    with pytest.raises(FileNotFoundError, match="validation-selected best.pt"):
        posthoc.run(tmp_path)


def test_list_runs_reports_actual_paths_status_and_commands_without_loading_models(tmp_path, monkeypatch, capsys):
    root = tmp_path / "checkpoints"
    ready = root / "idrid bs actual run"
    pending = root / "idrid_pending"
    seen = root / "idrid_all_data"
    for folder in [ready, pending, seen]:folder.mkdir(parents=True)
    (ready / "final.pt").write_bytes(b"listing must not load this")
    (seen / "final.pt").write_bytes(b"listing must not load this either")
    (pending / "last.pt").write_bytes(b"unfinished model")
    save_json(ready / "run_config.json", {"official_test_used_for_training": False})
    save_json(pending / "run_config.json", {"config": {"train": {"epochs": 500}}})
    save_json(pending / "metrics.json", {"epoch": 120})
    save_json(seen / "run_config.json", {"official_test_used_for_training": True})
    analysis = ready / "test_analysis";analysis.mkdir()
    save_json(analysis / "run_config.json", {"training_performed": False})
    def forbidden(*args, **kwargs):pytest.fail("Listing must not deserialize model weights.")
    monkeypatch.setattr(posthoc.torch, "load", forbidden)
    records = posthoc.list_runs(root)
    assert len(records) == 3
    by_path = {record["run_dir"]: record for record in records}
    assert str(ready) in by_path[str(ready)]["command"]
    assert "120/500" in by_path[str(pending)]["status"]
    assert "command" not in by_path[str(pending)] and "command" not in by_path[str(seen)]
    text = capsys.readouterr().out
    assert "FINAL FOUND" in text and "NO FINAL" in text and "TRAIN-SEEN" in text
