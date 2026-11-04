from __future__ import annotations

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

import idrid_seen_class1_tsne as analysis
from eyepacs_hybrid_ordinal.idrid import prepare_idrid
from smoke_test_idrid import write_synthetic_dataset


class Tiny(nn.Module):
    def __init__(self, predictions):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.register_buffer("scores", torch.tensor(predictions, dtype=torch.float32))

    def forward(self, images):
        features = torch.cat([images.mean((2, 3)), images.std((2, 3))], dim=1) * self.scale
        return {"pcol": features, "scol": features.square()+0.2, "prediction": self.scores[:len(images)]}


def setup_run(tmp_path, predictions):
    data, run = tmp_path / "data", tmp_path / "original_run"
    run.mkdir()
    write_synthetic_dataset(data / "raw")
    prepare_idrid(data / "raw", data / "prepared", strict=False)
    frames, _ = analysis.hybrid.load_prepared(data, allow_synthetic=True)
    cfg = {"seed": 42, "data": {"num_workers": 0, "batch_size": 10, "normalize": "imagenet", "image_size": 224},
           "model": {"num_classes": 5}}
    torch.save({"model": Tiny(predictions).state_dict(), "config": cfg, "epoch": 500,
                "official_test_used_for_training": True}, run / "final.pt")
    accuracy = float((predictions == frames["test"]["_label"].to_numpy()).mean())
    (run / "evaluation_metrics.json").write_text(json.dumps({"official_test_used_for_training": True, "seen_test": {"accuracy": accuracy}}))
    (run / "training_split.json").write_text(json.dumps({"train": frames["train"].sample_id.tolist()+frames["test"].sample_id.tolist()}))
    return run, data, frames


@pytest.mark.parametrize("all_c1_wrong", [False, True])
def test_fresh_inference_all_test_metrics_and_c1_all_other_correct(tmp_path, monkeypatch, all_c1_wrong):
    predictions = np.array([0, 1, 1, 0, 2, 4, 3, 1, 4, 2])
    if all_c1_wrong:
        predictions[2] = 0
    run, data, frames = setup_run(tmp_path, predictions)
    original = {p.name: p.read_bytes() for p in run.iterdir()}
    original_prepared = {p.name: p.read_bytes() for p in (data / "prepared").iterdir() if p.is_file()}
    monkeypatch.setattr(analysis, "build_evaluation_model", lambda cfg, device: Tiny(predictions).to(device))
    def forbidden(*args, **kwargs):
        pytest.fail("Analysis must not train or prepare data again.")
    monkeypatch.setattr(analysis.hybrid, "train_one_epoch", forbidden)
    monkeypatch.setattr("eyepacs_hybrid_ordinal.idrid.prepare_idrid", forbidden)
    from sklearn.decomposition import PCA
    fitted = []
    class RecordingPCA(PCA):
        def fit_transform(self, features, *args, **kwargs):
            fitted.append(features.copy())
            return super().fit_transform(features, *args, **kwargs)
    monkeypatch.setattr("sklearn.decomposition.PCA", RecordingPCA)
    out = tmp_path / "analysis"
    metrics, counts = analysis.analyze(run, data, out, allow_synthetic=True)
    labels = frames["test"]["_label"].to_numpy()
    selected = (labels == 1) | (predictions == labels)
    assert metrics["samples"] == 10
    assert metrics["accuracy"] == pytest.approx((predictions == labels).mean())
    assert metrics["quadratic_weighted_kappa"] == pytest.approx(cohen_kappa_score(labels, predictions, weights="quadratic", labels=list(range(5))))
    assert counts == {"0": 1, "1": 2, "2": 1, "3": 1, "4": 1}
    coords = pd.read_csv(out / "tsne_coordinates.csv")
    assert len(coords) == 6 and (coords.label == 1).sum() == 2
    assert (coords.loc[coords.label != 1, "label"] == coords.loc[coords.label != 1, "prediction_class"]).all()
    assert np.isfinite(coords[["tsne_x", "tsne_y"]]).all().all()
    with np.load(out / "test_embeddings.npz") as archive:
        assert len(archive["embeddings"]) == 10
        np.testing.assert_array_equal(archive["selected"], selected)
        np.testing.assert_array_equal(fitted[0], archive["embeddings"][selected])
    assert len(pd.read_csv(out / "test_predictions.csv")) == 10
    assert not (run / "predictions.csv").exists()  # Fresh predictions, no saved CSV required.
    report = json.loads((out / "evaluation_metrics.json").read_text())
    assert report["training_performed"] is False and report["evaluation_is_independent"] is False
    assert report["accuracy_difference_pp"] == 0
    assert report["checkpoint_epoch"] == 500
    meta = json.loads((out / "tsne_metadata.json").read_text())
    assert meta["included_incorrect_samples"] == (2 if all_c1_wrong else 1)
    assert meta["total_test_samples"] == 10 and meta["selected_samples"] == 6
    assert (out / "tsne_class1_all_other_correct.pdf").read_bytes().startswith(b"%PDF")
    assert (out / "tsne_class1_all_other_correct.png").stat().st_size > 1000
    assert {p.name: p.read_bytes() for p in run.iterdir()} == original
    assert {p.name: p.read_bytes() for p in (data / "prepared").iterdir() if p.is_file()} == original_prepared
    with pytest.raises(FileExistsError, match="new output folder"):
        analysis.analyze(run, data, out, allow_synthetic=True)


def test_rejects_train_only_result_before_loading_weights(tmp_path, monkeypatch):
    (tmp_path / "evaluation_metrics.json").write_text(json.dumps({"official_test_used_for_training": False}))
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("Must reject the wrong experiment before loading weights."))
    with pytest.raises(ValueError, match=r"original train\+test"):
        analysis.analyze(tmp_path, tmp_path, tmp_path / "out")


def test_cli_help_formats_and_defaults_point_to_the_existing_8252_model(capsys):
    with pytest.raises(SystemExit) as caught:
        analysis.parse_args(["--help"])
    assert caught.value.code == 0 and "--device" in capsys.readouterr().out
    args = analysis.parse_args([])
    assert args.run_dir == ROOT / "checkpoints/hybrid_idrid_all_data_gem_no_spatial_seed42"
    assert args.device == "cpu"
