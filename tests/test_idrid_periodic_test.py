from __future__ import annotations

import json
import random
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

import train_idrid_seen_balanced_softmax as runner
import train_idrid_all_data as all_data
from eyepacs_hybrid_ordinal.idrid import prepare_idrid
from smoke_test_idrid import write_synthetic_dataset
from test_idrid_seen_balanced_softmax import source, TinyBS


def test_short_budget_and_interval_are_explicit_without_changing_source(tmp_path):
    folder, original = source(tmp_path)
    cfg, saved = runner.configure(folder, tmp_path / "data", tmp_path / "out",
                                   epochs=200, test_every_epochs=100)
    assert saved == original and saved["train"]["epochs"] == 500
    assert cfg["train"]["epochs"] == 200 and cfg["train"]["test_every_epochs"] == 100
    assert cfg["provenance"]["epoch_budget"] == {"source": 500, "effective": 200, "changed": True}
    assert cfg["loss"]["margin_phase1_max_epochs"] == 100
    with pytest.raises(ValueError, match="Phase 2"):
        runner.configure(folder, tmp_path, tmp_path, epochs=100)
    with pytest.raises(ValueError, match="positive"):
        runner.configure(folder, tmp_path, tmp_path, test_every_steps=0)
    with pytest.raises(ValueError, match="Choose"):
        runner.configure(folder, tmp_path, tmp_path, test_every_steps=100, test_every_epochs=100)


@pytest.mark.parametrize("interval", ["epochs", "steps"])
def test_periodic_evaluation_is_at_requested_intervals_without_changing_training(tmp_path, monkeypatch, interval):
    torch.set_num_threads(2)
    data = tmp_path / "data"
    write_synthetic_dataset(data / "raw")
    prepare_idrid(data / "raw", data / "prepared", strict=False)
    folder, cfg = source(tmp_path)
    cfg["data"].update(batch_size=10, num_workers=0, train_batches_per_epoch=2)
    cfg["model"]["pretrained_imagenet"] = False
    cfg["train"].update(epochs=3, amp=False)
    cfg["loss"].update(margin_phase1_min_epochs=1, margin_phase1_max_epochs=1)
    (folder / "run_config.json").write_text(json.dumps({"config": cfg, "official_test_used_for_training": True}))
    import eyepacs_hybrid_ordinal.balanced_softmax as bs
    def model_factory(**kwargs):
        model = TinyBS(**kwargs)
        model.encoder.add_module("bn", nn.BatchNorm1d(4))
        model.encoder.add_module("dropout", nn.Dropout(0.3))
        return model
    monkeypatch.setattr(bs, "BalancedSoftmaxHybridOrdinalNet", model_factory)
    monkeypatch.setattr(runner.analysis, "analyze", lambda *args, **kwargs: None)
    original_evaluate = all_data.evaluate_seen_test
    calls = []
    def evaluate(model, *args, **kwargs):
        if kwargs.get("periodic"):
            assert model.training
            before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
            result = original_evaluate(model, *args, **kwargs)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)
            calls.append((kwargs["epoch"], kwargs["iteration"]))
            # The caller must isolate evaluation's RNG effects as well as BatchNorm.
            torch.rand(7); np.random.rand(7); random.random()
            return result
        return original_evaluate(model, *args, **kwargs)
    monkeypatch.setattr(all_data, "evaluate_seen_test", evaluate)
    common = dict(device="cpu", num_workers=0, allow_synthetic=True)
    reference = runner.run(folder, data, tmp_path / "reference", **common)
    options = {"test_every_epochs": 1} if interval == "epochs" else {"test_every_steps": 3}
    output = tmp_path / "periodic"
    result = runner.run(folder, data, output, **common, **options)
    expected = [(1, 2), (2, 4), (3, 6)] if interval == "epochs" else [(2, 3), (3, 6)]
    assert calls == expected
    a = torch.load(tmp_path / "reference/final.pt", weights_only=True)
    b = torch.load(output / "final.pt", weights_only=True)
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    torch.testing.assert_close(a["ordinal_margins"]["raw_margins"], b["ordinal_margins"]["raw_margins"], rtol=0, atol=0)
    assert reference["seen_test"] == result["seen_test"]
    assert result["periodic_test_evaluations"] == len(expected)
    table = pd.read_csv(output / "periodic_test_metrics.csv")
    assert list(zip(table.epoch, table.batch_iteration)) == expected
    assert table.official_test_used_for_training.all() and not table.evaluation_is_independent.any()
    history = json.loads((output / "periodic_test_metrics.json").read_text())
    assert history["used_for_scheduler_or_checkpoint_selection"] is False
    for row in table.to_dict("records"):
        directory = output / "periodic_test" / f"step_{row['batch_iteration']:08d}"
        payload = json.loads((directory / "evaluation_metrics.json").read_text())
        assert payload["checkpoint"] is None and payload["evaluation_kind"] == "periodic"
        assert payload["weights_source"] == "in_memory_after_iteration"
        assert payload["seen_test"]["samples"] == 10
        predictions = pd.read_csv(directory / "predictions.csv")
        assert len(predictions) == 10 and predictions.included_in_training_pool.all()
        assert row["accuracy_percent"] == pytest.approx(100*(predictions.label == predictions.prediction_class).mean())
        for grade in range(5):
            selected = predictions[predictions.label == grade]
            assert row[f"C{grade}_accuracy_percent"] == pytest.approx(100*(selected.label == selected.prediction_class).mean())
        assert not list(directory.glob("tsne*"))


def test_cli_exposes_epoch_budget_and_both_interval_units():
    args = runner.parse_args(["--epochs", "200", "--test-every-epochs", "100"])
    assert args.epochs == 200 and args.test_every_epochs == 100 and args.test_every_steps is None
    args = runner.parse_args(["--test-every-steps", "100"])
    assert args.test_every_steps == 100 and args.test_every_epochs is None
    with pytest.raises(SystemExit):
        runner.parse_args(["--test-every-epochs", "100", "--test-every-steps", "100"])
