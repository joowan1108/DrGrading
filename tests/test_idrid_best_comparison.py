from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import cohen_kappa_score
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import compare_idrid_best_eyepacs as comparison
import run_spatial_and_idrid_best as pipeline
import train_idrid as hybrid
from eyepacs_hybrid_ordinal.idrid import prepare_idrid
from eyepacs_hybrid_ordinal.idrid_analysis import save_correct_test_tsne
from smoke_test_idrid import write_synthetic_dataset


class TinyHybrid(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 4, 3, stride=8), nn.ReLU(),
                                     nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.pcol, self.scol, self.regression = nn.Linear(4, 8), nn.Linear(4, 8), nn.Linear(16, 1)
        with torch.no_grad():
            self.regression.weight.zero_()
            self.regression.bias.fill_(2.0)

    def forward(self, images):
        features = self.encoder(images)
        pcol, scol = self.pcol(features), self.scol(features)
        return {"pcol": pcol, "scol": scol,
                "prediction": self.regression(torch.cat([pcol, scol], dim=1)).squeeze(1)}


def test_source_snapshot_preserves_the_selected_eyepacs_settings(tmp_path):
    source, provenance = comparison.read_source()
    cfg = comparison.make_config(source, tmp_path / "data", tmp_path / "run")
    assert provenance["source_run"] == "hybrid_eyepacs_efficientnet_v2_s_384_gem_no_spatial"
    assert provenance["source_weights_loaded_for_idrid_training"] is False
    assert cfg["data"]["image_size"] == 384
    assert cfg["data"]["transform_profile"] == "eyepacs"
    assert cfg["data"]["batch_size"] == 24 and cfg["data"]["train_batches_per_epoch"] == 22
    assert cfg["loss"]["margin_lr"] == 1e-5
    assert cfg["loss"]["margin_phase1_min_epochs"] == 20
    assert cfg["loss"]["margin_phase1_max_epochs"] == 30
    assert cfg["loss"]["mmnp_pair_reduction"] == "sum"
    assert cfg["loss"]["classification_weight"] == 0
    assert cfg["model"]["balanced_softmax"] is False
    assert cfg["model"]["spatial_attention"] is False and cfg["model"]["pooling"] == "gem"
    assert cfg["train"]["epochs"] == 75 and cfg["train"]["early_stopping_patience"] is None
    assert cfg["train"]["checkpoint_selection"] == "final_epoch"
    assert source["train"]["early_stopping_patience"] == 13  # No source mutation.
    with pytest.raises(ValueError, match="Phase 1"):
        comparison.make_config(source, tmp_path, tmp_path, epochs=30)


@pytest.mark.parametrize("duplicate", [False, True])
def test_both_protocols_train_fresh_then_evaluate_all_test_and_only_plot_correct(tmp_path, monkeypatch, duplicate):
    torch.set_num_threads(2)
    data = tmp_path / "data"
    write_synthetic_dataset(data / "raw")
    if duplicate:
        original = data / "raw/B.%20Disease%20Grading/B. Disease Grading/1. Original Images"
        (original / "b. Testing Set/IDRiD_003.jpg").write_bytes(
            (original / "a. Training Set/IDRiD_001.jpg").read_bytes())
        with pytest.warns(UserWarning):
            prepare_idrid(data / "raw", data / "prepared", strict=False, overlap_policy="report")
    else:
        prepare_idrid(data / "raw", data / "prepared", strict=False)
    source, provenance = comparison.read_source()
    source["train"].update(epochs=2, amp=False)
    source["loss"].update(margin_phase1_min_epochs=1, margin_phase1_max_epochs=1)
    source["data"]["batch_size"] = 10
    original_loader, original_components = hybrid.make_loader, hybrid.build_components
    loader_calls, initial_states, step_counts = [], [], []

    def loader(frame, manifest, cfg, *, training):
        result = original_loader(frame, manifest, cfg, training=training)
        loader_calls.append((training, frame["sample_id"].tolist(), sum(step_counts)))
        # Both training and inference really use 384, rather than the legacy 224 loader.
        assert result.dataset[0][0].shape == (3, 384, 384)
        return result

    def components(cfg, device):
        result = original_components(cfg, device)
        assert result[2]["mmnp"].pair_reduction == "sum"
        initial_states.append({key: value.clone() for key, value in result[0].state_dict().items()})
        result[4].register_step_post_hook(lambda *args, **kwargs: step_counts.append(1))
        return result

    def forbidden(*args, **kwargs):
        pytest.fail("Must not prepare data, create validation splits or select checkpoints on test.")

    monkeypatch.setattr(hybrid, "HybridOrdinalNet", TinyHybrid)
    monkeypatch.setattr(hybrid, "make_loader", loader)
    monkeypatch.setattr(hybrid, "build_components", components)
    monkeypatch.setattr(hybrid, "selection_indices", forbidden)
    monkeypatch.setattr(hybrid, "evaluate", forbidden)
    monkeypatch.setattr("eyepacs_hybrid_ordinal.idrid.prepare_idrid", forbidden)
    output = tmp_path / "comparison"
    options = dict(data_dir=data, output_dir=output, device="cpu", num_workers=0,
                   batches_per_epoch=1, allow_synthetic=True)
    if duplicate:
        with pytest.warns(UserWarning, match="test images also occur"):
            results = comparison.run_comparison(source, provenance, **options)
    else:
        results = comparison.run_comparison(source, provenance, **options)
    assert len(initial_states) == 2 and len(step_counts) == 4
    for key in initial_states[0]:
        torch.testing.assert_close(initial_states[0][key], initial_states[1][key], rtol=0, atol=0)
    assert [call[0] for call in loader_calls] == [True, False, False, True, False, False]
    assert [call[2] for call in loader_calls if not call[0]] == [2, 2, 4, 4]
    table = pd.read_csv(output / "comparison.csv")
    assert table.training_samples.tolist() == [30, 20-int(duplicate)]
    assert table.test_samples.tolist() == [10, 10]
    assert table.official_test_used_for_training.tolist() == [True, False]
    assert table.evaluation_is_independent.tolist() == [False, True]
    assert json.loads((output / "comparison.json").read_text())["evaluation_status"] == "complete"
    for index, (name, seen) in enumerate(comparison.PROTOCOLS.items()):
        folder, result = output / name, results[name]
        split = json.loads((folder / "training_split.json").read_text())
        assert split["validation"] == []
        assert bool(set(split["train"]) & set(split["test"])) == seen
        assert len(split["official_test_used_for_training"]) == (10 if seen else 0)
        assert len(split["excluded_training_test_copies"]) == (int(duplicate) if not seen else 0)
        checkpoint = torch.load(folder / "final.pt", weights_only=True)
        assert checkpoint["official_test_used_for_training"] is seen
        assert checkpoint["margin_state"]["frozen"] is True
        assert checkpoint["optimizer_steps"] == result["optimizer_steps"] == 2
        assert not (folder / "best.pt").exists()
        predictions = pd.read_csv(folder / "test/predictions.csv")
        assert predictions.included_in_training_pool.tolist() == [seen]*10
        stats = result["test"]
        assert stats["samples"] == 10
        assert stats["accuracy"] == pytest.approx((predictions.label == predictions.prediction_class).mean())
        assert stats["quadratic_weighted_kappa"] == pytest.approx(cohen_kappa_score(
            predictions.label, predictions.prediction_class, labels=list(range(5)), weights="quadratic"))
        classes = pd.read_csv(folder / "test/per_class_metrics.csv")
        for grade in range(5):
            subset = predictions[predictions.label == grade]
            expected = 100*(subset.label == subset.prediction_class).mean()
            assert classes.iloc[grade].accuracy_percent == pytest.approx(expected)
            assert table.iloc[index][f"C{grade}_accuracy_percent"] == pytest.approx(expected)
        tsne = json.loads((folder / "test/tsne_test_correct_only_metadata.json").read_text())
        assert tsne["official_test_used_for_training"] is seen
        assert tsne["evaluation_is_independent"] is not seen
        with np.load(folder / "test/tsne_test_correct_only_embeddings.npz") as archive:
            assert (archive["labels"] == archive["predictions_class"]).all()
            expected_ids = predictions.loc[predictions.correct, "sample_id"].tolist()
            assert archive["sample_ids"].tolist() == expected_ids
    # Finished protocols can be reused, but never trained again or silently overwritten.
    monkeypatch.setattr(comparison, "fit_all", forbidden)
    if duplicate:
        with pytest.warns(UserWarning):
            rerun = comparison.run_comparison(source, provenance, skip_completed=True, **options)
    else:
        rerun = comparison.run_comparison(source, provenance, skip_completed=True, **options)
    assert rerun == results
    with pytest.raises(FileExistsError, match="new --output-dir"):
        comparison.run_comparison(source, provenance, **options)
    modified = copy.deepcopy(source); modified["loss"]["margin_lr"] = 1e-3
    with pytest.raises(FileExistsError):
        comparison.run_comparison(modified, provenance, skip_completed=True, **options)


@pytest.mark.parametrize("scope,seen", [("heldout_test", False), ("training_seen_test", True)])
def test_correct_only_actual_tsne_has_an_explicit_evaluation_scope(tmp_path, scope, seen):
    labels = np.array([0, 0, 1, 2, 3, 4])
    predictions = np.array([0, 2, 1, 2, 3, 4], dtype=float)
    metadata = save_correct_test_tsne(
        tmp_path, np.random.default_rng(4).normal(size=(6, 8)), predictions, labels,
        [f"image{i}" for i in range(6)], [f"test:image{i}" for i in range(6)], evaluation_scope=scope,
    )
    assert metadata["status"] == "created" and metadata["correct_samples"] == 5
    assert metadata["official_test_used_for_training"] is seen
    assert metadata["evaluation_is_independent"] is not seen
    assert (tmp_path / metadata["pdf_file"]).read_bytes().startswith(b"%PDF")


def test_pipeline_passes_the_exact_spatial_run_then_both_idrid_protocols(tmp_path, monkeypatch):
    prepared = tmp_path / "data/prepared"; prepared.mkdir(parents=True)
    (prepared / "manifest.json").write_text("{}")
    calls = []
    monkeypatch.setattr(pipeline.spatial, "run", lambda args: calls.append(("spatial", args)))
    monkeypatch.setattr(pipeline.comparison, "run_comparison", lambda *args, **kwargs: calls.append(("idrid", kwargs)))
    pipeline.main(["--data-dir", str(tmp_path / "data"), "--device", "cuda", "--output-dir", str(tmp_path / "output")])
    assert [name for name, _ in calls] == ["spatial", "idrid"]
    spatial = calls[0][1]
    assert spatial.run_dir.endswith("hybrid_eyepacs_efficientnet_v2_s_384_gem_spatial_cloc110-2")
    assert spatial.class_id == 3 and spatial.split == "val"
    assert calls[1][1]["data_dir"] == tmp_path / "data"
    assert calls[1][1]["output_dir"] == tmp_path / "output/idrid_comparison"
    assert calls[1][1]["epochs"] is None  # Inherit 75 from the pinned source, not the old 500.
