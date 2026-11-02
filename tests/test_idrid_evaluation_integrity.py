"""Regression tests for accidentally reporting seen images as held-out IDRiD."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import train_idrid as hybrid
import train_idrid_all_data as legacy
import train_idrid_fixed_iterations as heldout
from eyepacs_hybrid_ordinal.idrid import prepare_idrid
from smoke_test_idrid import smoke_config, write_synthetic_dataset
from report_idrid_test import report


def prepared(tmp_path):
    write_synthetic_dataset(tmp_path / "raw")
    prepare_idrid(tmp_path / "raw", tmp_path / "prepared", strict=False)
    return hybrid.load_prepared(tmp_path, allow_synthetic=True)


def test_legacy_train_plus_test_requires_explicit_diagnostic_opt_in():
    with pytest.raises(ValueError, match="legacy diagnostic trains on official TEST"):
        legacy.run({})  # Must fail before loading data or creating a checkpoint.


@pytest.mark.parametrize("state", ["training", "seen_test"])
def test_result_reporter_never_presents_training_as_heldout_test(tmp_path, state):
    if state == "seen_test":
        (tmp_path / "evaluation_metrics.json").write_text(json.dumps({"official_test_used_for_training": True, "seen_test": {"accuracy": 0.99}}))
    with pytest.raises(ValueError, match="NOT test accuracy|TRAIN-SEEN"):
        report(tmp_path)


def test_actual_images_are_rehashed_even_when_prepared_csv_has_not_changed(tmp_path):
    frames, manifest = prepared(tmp_path)
    root = Path(manifest["image_root"])
    train = root / frames["train"].iloc[0].image
    test = root / frames["test"].iloc[0].image
    train.write_bytes(test.read_bytes())
    with pytest.raises(ValueError, match="Image changed after preparation"):
        hybrid.load_prepared(tmp_path, allow_synthetic=True)


def test_decoded_pixel_copies_are_excluded_despite_different_file_bytes(tmp_path):
    frames, manifest = prepared(tmp_path)
    root = Path(manifest["image_root"])
    train_path = root / frames["train"].iloc[0].image
    test_path = root / frames["test"].iloc[2].image  # Preserve a different test label.
    with Image.open(train_path) as image:
        image.save(test_path, format="PNG")  # Same decoded RGB, different encoded bytes.
    assert train_path.read_bytes() != test_path.read_bytes()
    manifest = prepare_idrid(root, tmp_path / "prepared", strict=False, overlap_policy="report")
    assert manifest["duplicate_summary"]["overlapping_test_samples"] == 0
    with pytest.warns(UserWarning, match="1 test images also occur in training"):
        frames, manifest = hybrid.load_prepared(tmp_path, allow_synthetic=True)
    training, excluded = heldout.exclude_test_copies(frames["train"], frames["test"])
    assert len(training) == 19 and len(excluded) == 1
    assert len(frames["test"]) == 10 and frames["test"].iloc[2]._label == 1
    assert set(training._pixel_sha256).isdisjoint(frames["test"]._pixel_sha256)
    assert manifest["runtime_image_verification"]["overlapping_test_samples_file_or_pixels"] == 1


class ConstantPrediction(nn.Module):
    def forward(self, images):
        return {"prediction": torch.zeros(len(images), device=images.device)}


def test_test_evaluation_refuses_training_rows(tmp_path):
    frames, manifest = prepared(tmp_path)
    cfg = smoke_config(tmp_path, tmp_path / "results")
    with pytest.raises(ValueError, match="must use official TEST rows"):
        hybrid.evaluate_test(ConstantPrediction(), frames["train"], manifest, cfg, tmp_path,
                             torch.device("cpu"), False, training_frame=frames["train"], require_heldout=True)
    assert not (tmp_path / "evaluation_metrics.json").exists()


@pytest.mark.parametrize("column", ["sha256", "_pixel_sha256"])
def test_test_audit_checks_membership_instead_of_trusting_clean_flag(tmp_path, column):
    frames, manifest = prepared(tmp_path)
    test = frames["test"].copy()
    test.loc[test.index[0], column] = frames["train"].iloc[0][column]
    test["_overlaps_train"] = False
    cfg = smoke_config(tmp_path, tmp_path / "results")
    with pytest.raises(ValueError, match="Training/test overlap"):
        hybrid.evaluate_test(ConstantPrediction(), test, manifest, cfg, tmp_path,
                             torch.device("cpu"), False, training_frame=frames["train"], require_heldout=True)


def test_wrong_evaluation_loader_cannot_write_train_predictions_as_test(tmp_path, monkeypatch):
    frames, manifest = prepared(tmp_path)
    wrong = frames["train"].iloc[:10]
    monkeypatch.setattr(hybrid, "make_loader", lambda *args, **kwargs: [(
        torch.zeros(10, 3, 8, 8), torch.tensor(wrong._label.tolist()), wrong._image_id.tolist(),
    )])
    cfg = smoke_config(tmp_path, tmp_path / "results")
    with pytest.raises(ValueError, match="exactly the requested test images/labels"):
        hybrid.evaluate_test(ConstantPrediction(), frames["test"], manifest, cfg, tmp_path,
                             torch.device("cpu"), False, training_frame=frames["train"], require_heldout=True)
    assert not (tmp_path / "evaluation_metrics.json").exists()


def test_remote_wrapper_defaults_to_no_validation_heldout_runner(tmp_path):
    calls = tmp_path / "calls.txt"
    fake = tmp_path / "fake_python"
    fake.write_text(f'#!{sys.executable}\nimport json,sys\nwith open({str(calls)!r}, "a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n')
    fake.chmod(0o755)
    env = dict(os.environ, IDRID_PYTHON=str(fake), IDRID_DATA_DIR=str(tmp_path / "data with spaces"))
    subprocess.run(["bash", "scripts/run_idrid_remote.sh", "--device", "cuda"], cwd=ROOT, env=env, check=True)
    records = [json.loads(line) for line in calls.read_text().splitlines()]
    assert records[0][0] == "scripts/prepare_idrid.py"
    assert records[0][-2:] == ["--overlap-policy", "report"]
    assert records[1] == ["scripts/train_idrid_fixed_iterations.py", "--data-dir", str(tmp_path / "data with spaces"), "--device", "cuda"]
