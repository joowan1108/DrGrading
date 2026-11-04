from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from export_spatial_attention import crop_box, parse_args, run, select_sample
from eyepacs_hybrid_ordinal.data import load_eyepacs_dataframe
from eyepacs_hybrid_ordinal.models import HybridOrdinalNet
from eyepacs_hybrid_ordinal.models import SpatialAttention
from eyepacs_hybrid_ordinal.splitting import build_nested_split_indices
from eyepacs_hybrid_ordinal.transforms import CropBlackBorder


@pytest.fixture
def experiment(tmp_path):
    torch.set_num_threads(1)
    images = tmp_path / "images"
    images.mkdir()
    rows = []
    for subject in range(60):
        for side in ["left", "right"]:
            image_id = f"{subject:03d}_{side}"
            image = Image.new("RGB", (95, 77), "black")
            ImageDraw.Draw(image).ellipse((18, 14, 77, 63), fill=(90, 45, 20))
            image.save(images / f"{image_id}.png")
            rows.append({"image": image_id, "level": subject % 5})
    labels = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(labels, index=False)
    cfg = {
        "seed": 42,
        "data": {"labels_csv": str(labels), "image_root": str(images), "image_size": 64, "normalize": "imagenet"},
        "split": {"num_folds": 3, "fold_index": 0, "val_size": 0.4, "subject_independent": True},
        "model": {"backbone": "efficientnet_b0", "projection_hidden_dim": 16, "projection_dim": 8,
                  "regression_hidden_dim": 16, "regression_input": "projection_concat", "dropout": 0.0,
                  "spatial_attention": True, "spatial_attention_kernel_size": 7, "pooling": "gem", "num_classes": 5},
    }
    model = HybridOrdinalNet(
        backbone="efficientnet_b0", projection_hidden_dim=16, projection_dim=8,
        regression_hidden_dim=16, regression_input="projection_concat", dropout=0,
        spatial_attention=True, spatial_attention_kernel_size=7, pooling="gem",
    )
    # Known checkpoint parameters make the exact sigmoid gate 0.5 everywhere.
    # Force an incorrect prediction to verify selection uses the true class.
    with torch.no_grad():
        model.spatial_attention.conv.weight.zero_()
        model.regression_head.net[-1].weight.zero_()
        model.regression_head.net[-1].bias.fill_(2.0)
    checkpoint = tmp_path / "best.pt"
    torch.save({"model": model.state_dict(), "config": cfg, "epoch": 9}, checkpoint)
    return checkpoint, cfg, tmp_path


def test_exports_checkpoint_gate_and_original_geometry_for_one_true_class3(experiment):
    checkpoint, cfg, tmp_path = experiment
    output = tmp_path / "maps"
    args = parse_args(["--checkpoint", str(checkpoint), "--device", "cpu", "--output-dir", str(output)])
    report = run(args)
    assert report["ground_truth_class"] == 3
    assert report["predicted_class"] == 2
    assert report["correct"] is False
    assert report["checkpoint_epoch"] == 9
    native = np.load(output / "spatial_attention.npy")
    enlarged = np.load(output / "spatial_attention_input.npy")
    np.testing.assert_array_equal(native, np.full((2, 2), 0.5))
    np.testing.assert_allclose(enlarged, 0.5, atol=1e-7)
    assert report["display_limits"] == [0.0, 1.0]
    assert Image.open(output / "input.png").size == (64, 64)
    with Image.open(report["image_path"]) as original:
        box = crop_box(original)
        np.testing.assert_array_equal(np.asarray(original.crop(box)), np.asarray(CropBlackBorder()(original)))
        overlay = Image.open(output / "overlay_original.png")
        assert overlay.size == original.size
        assert overlay.getpixel((0, 0)) == original.getpixel((0, 0))
    for filename in report["outputs"]:
        assert (output / filename).stat().st_size > 0
    assert json.loads((output / "metadata.json").read_text())["image_id"] == report["image_id"]


def test_selection_respects_saved_subject_splits(experiment):
    _, cfg, _ = experiment
    frame = load_eyepacs_dataframe(cfg["data"]["labels_csv"])
    train, val, test = build_nested_split_indices(frame, seed=42, val_size=0.4, num_folds=3,
                                                  fold_index=0, subject_independent=True)
    subjects = [set(frame.iloc[idx]["_subject_id"]) for idx in [train, val, test]]
    assert not subjects[0] & subjects[1] and not subjects[0] & subjects[2] and not subjects[1] & subjects[2]
    for split, expected in [("val", val), ("test", test)]:
        row, _, _ = select_sample(cfg, split=split, class_id=3)
        assert row["_label"] == 3
        assert row["_image_id"] in set(frame.iloc[expected]["_image_id"])
        assert row["_subject_id"] not in subjects[0]
    row, _, _ = select_sample(cfg, class_id=2)
    with pytest.raises(ValueError, match="not a ground-truth Class 3"):
        select_sample(cfg, class_id=3, image_id=row["_image_id"])


def test_rejects_no_spatial_checkpoint_without_exporting_random_maps(experiment):
    checkpoint, cfg, tmp_path = experiment
    payload = torch.load(checkpoint, weights_only=True)
    del payload["model"]["spatial_attention.conv.weight"]
    cfg["model"]["spatial_attention"] = False
    payload["config"] = cfg
    no_spatial = tmp_path / "no_spatial.pt"
    torch.save(payload, no_spatial)
    output = tmp_path / "should_not_exist"
    with pytest.raises(ValueError, match="no trained Spatial Attention weights"):
        run(parse_args(["--checkpoint", str(no_spatial), "--device", "cpu", "--output-dir", str(output)]))
    assert not output.exists()
    with pytest.raises(ValueError) as caught:
        run(parse_args(["--checkpoint", str(no_spatial)]))
    assert "spatial_attention=False" in str(caught.value)
    assert str(no_spatial) in str(caught.value)
    assert "--list-runs" in str(caught.value)


def test_run_dir_and_explicit_image_selection(experiment):
    checkpoint, cfg, tmp_path = experiment
    row, _, _ = select_sample(cfg, class_id=3, sample_index=1)
    args = parse_args(["--run-dir", str(checkpoint.parent), "--image-id", row["_image_id"],
                       "--device", "cpu", "--output-dir", str(tmp_path / "chosen")])
    report = run(args)
    assert report["image_id"] == row["_image_id"]
    assert report["selection"] == "explicit image id"


def test_rejects_test_selection_when_training_has_no_outer_test(experiment):
    _, cfg, _ = experiment
    cfg["split"]["num_folds"] = 0
    with pytest.raises(ValueError, match="no configured outer test split"):
        select_sample(cfg, split="test", class_id=3)


def test_missing_checkpoint_explains_expected_spatial_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="actual EyePACS Spatial Attention"):
        run(parse_args(["--run-dir", str(tmp_path / "missing_run")]))


def test_exact_run_is_found_in_sibling_and_data_paths_follow_checkpoint(experiment, monkeypatch):
    import export_spatial_attention as exporter

    checkpoint, cfg, tmp_path = experiment
    idrid_root, eyepacs_root = tmp_path / "DrGrading-idrid", tmp_path / "DrGrading"
    idrid_root.mkdir()
    run_dir = eyepacs_root / "checkpoints/hybrid_eyepacs_efficientnet_v2_s_e1_e2_gem384"
    run_dir.mkdir(parents=True)
    payload = torch.load(checkpoint, weights_only=True)
    payload["config"]["data"].update(labels_csv="data/eyepacs/trainLabels.csv", image_root="data/eyepacs/train")
    torch.save(payload, run_dir / "best.pt")
    monkeypatch.setattr(exporter, "ROOT", idrid_root)
    args = parse_args(["--run-dir", f"checkpoints/{run_dir.name}"])
    path, _, _, resolved = exporter.checkpoint_and_config(args)
    assert path == run_dir / "best.pt"
    assert Path(resolved["data"]["labels_csv"]) == eyepacs_root / "data/eyepacs/trainLabels.csv"
    assert Path(resolved["data"]["image_root"]) == eyepacs_root / "data/eyepacs/train"
    other = tmp_path / "other/checkpoints" / run_dir.name
    other.mkdir(parents=True)
    with pytest.raises(ValueError, match="Several folders"):
        exporter.resolve_run_directory(f"checkpoints/{run_dir.name}")


def test_default_spatial_run_matches_archived_training_settings():
    import export_spatial_attention as exporter
    import run_spatial_and_idrid_best as pipeline

    # Real saved run metadata: the misleading name must never become the default again.
    misleading = "hybrid_eyepacs_efficientnet_v2_s_384_gem_spatial_cloc110-2"
    with tarfile.open(ROOT / "checkpoints.tar.gz") as archive:
        wrong = json.load(archive.extractfile(f"checkpoints/{misleading}/run_config.json"))
        correct = json.load(archive.extractfile(f"checkpoints/{exporter.DEFAULT_SPATIAL_RUN}/run_config.json"))
    assert wrong["config"]["model"]["spatial_attention"] is False
    assert correct["config"]["model"]["spatial_attention"] is True
    assert pipeline.parse_args([]).spatial_run == f"checkpoints/{exporter.DEFAULT_SPATIAL_RUN}"


def test_listing_checks_weights_and_settings_and_reports_real_inference_commands(experiment, monkeypatch, capsys):
    import export_spatial_attention as exporter

    checkpoint, cfg, tmp_path = experiment
    root = tmp_path / "projects"
    ready = root / "actual/checkpoints/e1_e2 real run"
    misleading = root / "idrid/checkpoints/gem_spatial_cloc110-2"
    missing = root / "idrid/checkpoints/spatial_missing_best"
    disabled = root / "idrid/checkpoints/disabled"
    unreadable = root / "idrid/checkpoints/unreadable"
    for folder in [ready, misleading, missing, disabled, unreadable]:
        folder.mkdir(parents=True)
    payload = torch.load(checkpoint, weights_only=True)
    torch.save(payload, ready / "best.pt")
    cfg["model"]["spatial_attention"] = True
    (missing / "run_config.json").write_text(json.dumps({"config": cfg}))
    payload["config"]["model"]["spatial_attention"] = False
    torch.save(payload, disabled / "best.pt")  # Config/weights disagree: never list as READY.
    del payload["model"]["spatial_attention.conv.weight"]
    torch.save(payload, misleading / "best.pt")
    (unreadable / "best.pt").write_bytes(b"not a torch checkpoint")
    monkeypatch.setattr(exporter, "ROOT", root / "idrid")

    def forbidden(*args, **kwargs):
        pytest.fail("Listing must not instantiate a network, read images or run inference.")

    monkeypatch.setattr(exporter, "HybridOrdinalNet", forbidden)
    monkeypatch.setattr(exporter, "select_sample", forbidden)
    records = exporter.main(["--list-runs"])
    by_dir = {Path(row["run_dir"]): row for row in records}
    assert by_dir[ready]["status"] == "READY"
    assert by_dir[ready]["spatial_weight_shape"] == [1, 2, 7, 7]
    assert f"'{ready / 'best.pt'}'" in by_dir[ready]["command"]  # Safe command for paths with spaces.
    assert by_dir[misleading]["status"] == "NO SPATIAL WEIGHTS"
    assert by_dir[misleading]["spatial_attention_config"] is False
    assert by_dir[missing]["status"] == "BEST MISSING"
    assert by_dir[disabled]["status"] == "DISABLED IN SAVED CONFIG"
    assert by_dir[unreadable]["status"] == "INSPECTION FAILED"
    assert sum("command" in row for row in records) == 1
    assert "NO SPATIAL WEIGHTS" in capsys.readouterr().out
    assert len(exporter.main(["--list-runs", "--search-root", str(ready)])) == 1


def test_exports_nonconstant_sigmoid_values_without_rescaling_saved_array(experiment, monkeypatch):
    import export_spatial_attention as exporter

    _, cfg, tmp_path = experiment

    class KnownSpatialModel(torch.nn.Module):
        def __init__(self, **_kwargs):
            super().__init__()
            self.spatial_attention = SpatialAttention(7)
            self.register_buffer("pattern", torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2))
            with torch.no_grad():
                self.spatial_attention.conv.weight.zero_()
                self.spatial_attention.conv.weight[0, 0, 3, 3] = 0.2
                self.spatial_attention.conv.weight[0, 1, 3, 3] = -0.1

        def forward(self, images):
            weighted = self.spatial_attention(self.pattern.expand(len(images), -1, -1, -1))
            return {"prediction": weighted.sum(dim=(1, 2, 3))*0+3}

    model = KnownSpatialModel()
    checkpoint = tmp_path / "known_gate.pt"
    torch.save({"model": model.state_dict(), "config": cfg, "epoch": 7}, checkpoint)
    monkeypatch.setattr(exporter, "HybridOrdinalNet", KnownSpatialModel)
    output = tmp_path / "nonconstant"
    report = run(parse_args(["--checkpoint", str(checkpoint), "--device", "cpu", "--output-dir", str(output)]))
    expected = torch.tensor([[0, 0.1], [0.2, 0.3]]).sigmoid().numpy()
    np.testing.assert_allclose(np.load(output / "spatial_attention.npy"), expected, atol=1e-7)
    np.testing.assert_allclose(report["display_limits"], [float(expected.min()), float(expected.max())], atol=1e-7)
