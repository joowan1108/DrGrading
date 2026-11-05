from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import compare_gap_gem_maps as compare
from eyepacs_hybrid_ordinal.idrid import digest, pixel_digest
from eyepacs_hybrid_ordinal.models import GeMPooling, HybridOrdinalNet
from eyepacs_hybrid_ordinal.pooling_visualization import (
    lesion_map_metrics, pooling_moment_map, positive_unit_map, regression_maps,
)


class AnalyticModel(nn.Module):
    """Identity features and a single pooling output with a known derivative."""
    def __init__(self, exponent=None, sign=1):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.features = nn.Identity()
        self.global_pool = GeMPooling(exponent) if exponent else None
        self.sign = sign

    def forward(self, x):
        features = self.encoder.features(x)
        pooled = features.mean((-2, -1)) if self.global_pool is None else self.global_pool(features)
        return {"prediction": self.sign * pooled.sum()}


@pytest.mark.parametrize("exponent", [None, 3.0])
def test_known_pooling_derivative_and_unchanged_weights(exponent):
    image = torch.tensor([[[1., 1.], [1., 8.]]])
    model = AnalyticModel(exponent).eval().requires_grad_(False)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    score, maps = regression_maps(model, image)
    p = exponent or 1.
    assert score == pytest.approx(float((image.pow(p).mean()).pow(1/p)))
    # Local x*dy/dx preserves the powers, while Grad-CAM averages dy/dx spatially.
    local = maps["gradient_activation"]
    np.testing.assert_allclose(local/local.sum(), (image.pow(p)/image.pow(p).sum())[0].numpy(), rtol=1e-6)
    np.testing.assert_allclose(maps["gradcam"]/maps["gradcam"].sum(), (image/image.sum())[0].numpy(), rtol=1e-6)
    assert not model.encoder.features._forward_hooks
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    assert all(parameter.grad is None for parameter in model.parameters())
    if exponent:
        assert maps["moment_gem"][1, 1] == pytest.approx(512/515)
        assert maps["moment_gap"][1, 1] == pytest.approx(8/11)


def test_negative_and_zero_evidence_stay_zero():
    model = AnalyticModel(sign=-1).eval().requires_grad_(False)
    _, maps = regression_maps(model, torch.ones(1, 2, 2))
    assert (maps["gradcam_signed"] < 0).all()
    np.testing.assert_array_equal(positive_unit_map(maps["gradcam"]), np.zeros((2, 2)))
    info = lesion_map_metrics(maps["gradcam"], np.ones((6, 8), bool), (1, 1, 7, 5))
    assert info["has_positive_mass"] is False
    assert info["lesion_mass_fraction"] is None and info["lesion_area_enrichment"] is None


def test_hooks_removed_after_invalid_output():
    model = AnalyticModel(sign=float("nan")).eval().requires_grad_(False)
    with pytest.raises(ValueError, match="finite regression"):
        regression_maps(model, torch.ones(1, 2, 2))
    assert not model.encoder.features._forward_hooks


def test_stable_moments_and_crop_mask_metrics():
    features = torch.tensor([[[0., 1.], [2., 1000.]], [[0., 0.], [0., 0.]]])
    moment = pooling_moment_map(features, 10000)
    assert np.isfinite(moment).all() and float(moment.sum()) == pytest.approx(1)
    np.testing.assert_allclose(moment, [[.125, .125], [.125, .625]])
    union = np.zeros((8, 9), bool)
    union[2:6, 3:7] = True
    info = lesion_map_metrics(np.ones((2, 2)), union, (1, 1, 9, 7))
    assert info["lesion_area_fraction"] == pytest.approx(16/48)
    assert info["lesion_mass_fraction"] == pytest.approx(16/48)
    assert info["lesion_area_enrichment"] == pytest.approx(1)
    assert lesion_map_metrics(np.ones((2, 2)), None, (0, 0, 2, 2))["lesion_mass_fraction"] is None


def test_overlay_opacity_is_continuous_at_zero_and_preserves_data():
    original = Image.new("RGB", (5, 1), (100, 100, 100))
    values = np.array([[0, 1e-7, .25, .5, 1]], dtype=np.float32)
    before = values.copy()
    def red_map(unit):
        rgba = np.zeros((*unit.shape, 4), np.float32)
        rgba[..., 0] = 200/255
        rgba[..., 3] = 1
        return rgba
    result = compare.overlay(original, values, (0, 0, 5, 1), .4, red_map, 1.)
    np.testing.assert_array_equal(np.asarray(result)[0],
                                  [[100, 100, 100], [100, 100, 100], [110, 90, 90], [120, 80, 80], [140, 60, 60]])
    np.testing.assert_array_equal(values, before)
    np.testing.assert_array_equal(np.asarray(original), np.full((1, 5, 3), 100, np.uint8))


def test_continuous_overlay_does_not_change_pixels_outside_inference_crop():
    original = Image.new("RGB", (8, 6), (80, 90, 100))
    def white_map(unit):
        return np.ones((*unit.shape, 4), np.float32)
    result = np.asarray(compare.overlay(original, np.ones((2, 2)), (2, 1, 6, 5), .5, white_map, 1.))
    outside = np.ones((6, 8), bool)
    outside[1:5, 2:6] = False
    np.testing.assert_array_equal(result[outside], np.asarray(original)[outside])
    assert (result[1:5, 2:6] > np.asarray(original)[1:5, 2:6]).all()


@pytest.fixture
def pair(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    cfg = {"seed": 42, "data": {"image_size": 64, "normalize": "imagenet"},
           "split": {"num_folds": 3, "fold_index": 0, "val_size": .2, "subject_independent": True},
           "model": {"backbone": "efficientnet_b0", "projection_hidden_dim": 16, "projection_dim": 8,
                     "regression_hidden_dim": 16, "regression_input": "projection_concat", "dropout": 0,
                     "pooling": "avg", "spatial_attention": False, "num_classes": 5},
           "loss": {"alpha": .1, "beta": .1}, "train": {"epochs": 75, "output_dir": "ignored"}}
    paths = {}
    for role, pooling in [("gap", "avg"), ("gem", "gem")]:
        saved = copy.deepcopy(cfg)
        saved["model"]["pooling"] = pooling
        model = HybridOrdinalNet(**{k: v for k, v in saved["model"].items() if k != "num_classes"})
        with torch.no_grad():
            model.regression_head.net[-1].bias.fill_(2 if role == "gap" else 3)
        folder = tmp_path / role
        folder.mkdir()
        path = folder / "best.pt"
        torch.save({"config": saved, "model": model.state_dict(), "epoch": 53 if role == "gap" else 59,
                    "val_metrics": {"accuracy": .7959 if role == "gap" else .8145}}, path)
        paths[role] = path
    image = np.zeros((80, 100, 3), np.uint8)
    image[12:68, 20:80] = [140, 60, 30]
    image[30:35, 35:40] = [200, 80, 40]
    original = tmp_path / "original.png"
    Image.fromarray(image).save(original)
    mask = np.zeros((80, 100), np.uint8)
    mask[30:33, 35:38] = 255
    mask_path = tmp_path / "original_HE.tif"
    Image.fromarray(mask).save(mask_path)
    return paths, cfg, original, mask_path


def test_real_model_export_with_boxes_predictions_and_provenance(pair, tmp_path, monkeypatch):
    paths, _, image, mask = pair
    def no_download(*args, **kwargs):
        pytest.fail("Inference must not download pretrained weights.")
    monkeypatch.setattr(torch.hub, "download_url_to_file", no_download)
    before = {role: digest(path) for role, path in paths.items()}
    out = tmp_path / "maps"
    result = compare.run(compare.parse_args([
        "--gap-checkpoint", str(paths["gap"]), "--gem-checkpoint", str(paths["gem"]),
        "--image", str(image), "--mask", str(mask), "--true-class", "3", "--device", "cpu",
        "--output-dir", str(out),
    ]))
    assert result["controlled_pooling_comparison"] and not result["training_performed"]
    case = result["cases"][0]
    assert case["models"]["gap"]["predicted_class"] == 2
    assert case["models"]["gem"]["predicted_class"] == 3
    assert not case["models"]["gap"]["correct"] and case["models"]["gem"]["correct"]
    assert case["original_size"] == [100, 80] and len(case["boxes"]) == 1
    assert case["crop_box_xyxy"] == [12, 4, 88, 76]
    for role in ["gap", "gem"]:
        assert digest(paths[role]) == before[role]
        assert result["models"][role]["validation_accuracy_percent"] == pytest.approx(79.59 if role == "gap" else 81.45)
        with np.load(Path(case["directory"]) / f"{role}_maps.npz") as saved:
            assert saved["gradcam"].shape == (2, 2)
            assert np.isfinite(saved["gradient_activation"]).all()
    with np.load(Path(case["directory"]) / "gem_maps.npz") as saved:
        for name in ["moment_gap", "moment_gem"]:
            assert float(saved[name].sum()) == pytest.approx(1)
            assert 0 < case["models"]["gem"]["maps"][name]["lesion_area_fraction"] < 1
    assert result["models"]["gem"]["sha256"] == before["gem"]
    assert json.loads((out / "comparison_metadata.json").read_text())["cases"][0]["ground_truth_class"] == 3
    for filename in result["outputs"]:
        assert Path(filename).stat().st_size > 0
    with pytest.raises(FileExistsError, match="automatically dated"):
        compare.run(compare.parse_args(["--gap-checkpoint", str(paths["gap"]), "--gem-checkpoint", str(paths["gem"]),
                                        "--image", str(image), "--output-dir", str(out)]))
    # Re-rendering must work without any checkpoint access, training, or inference.
    def no_weights(*args, **kwargs):
        pytest.fail("Cached replot must not load model weights or run inference.")
    monkeypatch.setattr(compare, "read_checkpoint", no_weights)
    monkeypatch.setattr(compare, "infer_cases", no_weights)
    next_out = tmp_path / "continuous_replot"
    source_before = digest(out / "comparison_metadata.json")
    replotted = compare.run(compare.parse_args(["--replot-dir", str(out), "--output-dir", str(next_out)]))
    assert replotted["rendering"]["mode"] == "continuous_alpha"
    assert replotted["native_maps_unchanged"] and not replotted["inference_performed"]
    assert replotted["cases"][0]["models"] == case["models"]
    for role in ["gap", "gem"]:
        old_map = Path(case["directory"]) / f"{role}_maps.npz"
        new_map = Path(replotted["cases"][0]["directory"]) / f"{role}_maps.npz"
        assert old_map.read_bytes() == new_map.read_bytes()
    assert (out / "comparison_metrics.csv").read_bytes() == (next_out / "comparison_metrics.csv").read_bytes()
    assert digest(out / "comparison_metadata.json") == source_before


@pytest.mark.parametrize("change, message", [
    ("pool", "Expected a trained"), ("missing_p", "disagree"),
    ("spatial", "Spatial Attention"), ("bs", "Balanced-Softmax"), ("missing_weight", "state_dict"),
])
def test_invalid_checkpoints_cannot_export_initialized_maps(pair, change, message, tmp_path):
    paths, _, _, _ = pair
    payload = torch.load(paths["gem"], weights_only=True)
    if change == "pool":
        with pytest.raises(ValueError, match=message):
            compare.read_checkpoint(paths["gem"], "gap")
        return
    if change == "missing_p":
        del payload["model"]["global_pool.p"]
    if change == "spatial":
        payload["config"]["model"]["spatial_attention"] = True
    if change == "bs":
        payload["model"]["classification_head.fc.weight"] = torch.zeros(5, 16)
    if change == "missing_weight":
        del payload["model"]["regression_head.net.3.bias"]
    bad = tmp_path / "bad.pt"
    torch.save(payload, bad)
    if change == "missing_weight":
        cfg, state, _ = compare.read_checkpoint(bad, "gem")
        with pytest.raises(RuntimeError, match=message):
            compare.build_model(cfg, state, torch.device("cpu"))
    else:
        with pytest.raises(ValueError, match=message):
            compare.read_checkpoint(bad, "gem")


def test_controls_and_previous_metadata_preserve_dataset_and_pixels(pair, tmp_path):
    paths, _, image, mask = pair
    gap, _, _ = compare.read_checkpoint(paths["gap"], "gap")
    gem, _, _ = compare.read_checkpoint(paths["gem"], "gem")
    assert not compare.validate_pair(gap, gem)
    different = copy.deepcopy(gem)
    different["loss"]["alpha"] = .5
    with pytest.raises(ValueError, match="loss.alpha"):
        compare.validate_pair(gap, different)
    assert compare.validate_pair(gap, different, True)
    different["data"]["image_size"] = 384
    with pytest.raises(ValueError, match="different input"):
        compare.validate_pair(gap, different, True)
    metadata = tmp_path / "lesion_box_metadata.json"
    metadata.write_text(json.dumps({"image_path": str(image), "image_pixel_sha256": pixel_digest(image),
                                     "ground_truth_class": 2, "dataset": "IDRiD segmentation",
                                     "masks": [{"path": str(mask), "sha256": digest(mask)}]}))
    args = compare.parse_args(["--case-metadata", str(metadata)])
    selected = compare.select_cases(args, gem)[0]
    assert selected["image_path"] == str(image) and selected["ground_truth_class"] == 2
    assert selected["selection"]["dataset"] == "IDRiD segmentation"
    metadata.write_text(metadata.read_text().replace(pixel_digest(image), "not-the-same-pixels"))
    with pytest.raises(ValueError, match="pixels no longer match"):
        compare.select_cases(args, gem)


def test_cli_help_and_trained_run_discovery(pair, capsys):
    with pytest.raises(SystemExit) as caught:
        compare.parse_args(["--help"])
    assert caught.value.code == 0
    paths, _, _, _ = pair
    found = compare.list_runs(paths["gap"].parent.parent)
    assert {record["pooling"] for record in found} == {"gap", "gem"}
    assert "79.59%" in capsys.readouterr().out


def test_exact_sibling_run_resolution_never_substitutes_another_checkpoint(tmp_path, monkeypatch):
    current = tmp_path / "DrGrading-idrid"
    current.mkdir()
    folder = tmp_path / "DrGrading/checkpoints" / compare.GAP_RUN
    folder.mkdir(parents=True)
    best = folder / "best.pt"
    best.write_bytes(b"path-resolution-only fixture")
    monkeypatch.setattr(compare, "ROOT", current)
    assert compare.resolve_checkpoint(Path("checkpoints") / compare.GAP_RUN) == best
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        compare.resolve_checkpoint(Path("checkpoints") / compare.GAP_RUN / "final.pt")
    other = tmp_path / "other/checkpoints" / compare.GAP_RUN
    other.mkdir(parents=True)
    (other / "best.pt").write_bytes(b"another path fixture")
    with pytest.raises(FileNotFoundError, match="found 2"):
        compare.resolve_checkpoint(Path("checkpoints") / compare.GAP_RUN)


def test_latest_replot_source_uses_complete_metadata_and_rejects_missing_raw_maps(tmp_path, pair, monkeypatch):
    import os
    monkeypatch.setattr(compare, "ROOT", tmp_path)
    for name, stamp in [("older", 10), ("newer", 20)]:
        folder = tmp_path / "outputs" / name
        folder.mkdir(parents=True)
        metadata = folder / "comparison_metadata.json"
        metadata.write_text(json.dumps({"cases": [{"placeholder": True}]}))
        os.utime(metadata, (stamp, stamp))
    assert compare.resolve_replot_metadata(compare.parse_args(["--replot-latest"])).parent.name == "newer"
    broken = tmp_path / "outputs/broken"
    broken.mkdir()
    (broken / "comparison_metadata.json").write_text("incomplete JSON")
    assert compare.resolve_replot_metadata(compare.parse_args(["--replot-latest"])).parent.name == "newer"
    _, _, image, _ = pair
    source = tmp_path / "outputs/newer"
    (source / "comparison_metadata.json").write_text(json.dumps({"cases": [
        {"directory": str(source / "case_01_original"), "image_path": str(image)}]}))
    output = tmp_path / "must_not_be_created"
    with pytest.raises(FileNotFoundError, match="PNG files alone are insufficient"):
        compare.run(compare.parse_args(["--replot-dir", str(source), "--output-dir", str(output)]))
    assert not output.exists()
