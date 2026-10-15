"""Offline two-epoch all-data/seen-test smoke test with a real EfficientNetV2-S."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch

from smoke_test_idrid import ROOT, smoke_config, write_synthetic_dataset
from eyepacs_hybrid_ordinal.idrid import prepare_idrid
from train_idrid_all_data import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()
    work = args.work_dir or Path(tempfile.mkdtemp(prefix="idrid-all-data-smoke-"))
    if work.exists() and any(work.iterdir()):
        parser.error("--work-dir must be empty")
    torch.set_num_threads(2)
    write_synthetic_dataset(work / "data" / "raw")
    prepare_idrid(work / "data" / "raw", work / "data" / "prepared", strict=False)
    cfg = smoke_config(work / "data", work / "results", ROOT / "configs/hybrid_idrid_all_data_no_spatial.yaml")
    cfg["split"]["val_size"] = 0.0
    payload = run(cfg, allow_synthetic=True)
    split = json.loads((work / "results" / "training_split.json").read_text())
    history = json.loads((work / "results" / "metrics_history.json").read_text())["epochs"]
    assert len(split["train"]) == 30 and split["validation"] == []
    assert [row["phase"] for row in history] == ["phase1_joint", "phase2_frozen"]
    assert payload["seen_test"]["samples"] == 10
    assert payload["evaluation_is_independent"] is False
    assert payload["final_epoch"] == 2
    assert cfg["model"]["pooling"] == "gem" and cfg["model"]["spatial_attention"] is False
    print(
        "PASS: real EfficientNetV2-S + GeM, no Spatial Attention, all-data training, "
        "no validation, both phases, final checkpoint reload, SAME-test resubstitution Accuracy/MAE.\n"
        f"Synthetic artifacts: {work}"
    )


if __name__ == "__main__":
    main()
