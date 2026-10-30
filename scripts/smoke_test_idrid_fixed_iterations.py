"""Offline real-EfficientNet test of fixed-iteration TRAIN-only / held-out TEST."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch

from smoke_test_idrid import ROOT, smoke_config, write_synthetic_dataset
from eyepacs_hybrid_ordinal.idrid import prepare_idrid
from train_idrid_fixed_iterations import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()
    work = args.work_dir or Path(tempfile.mkdtemp(prefix="idrid-heldout-smoke-"))
    if work.exists() and any(work.iterdir()):
        parser.error("--work-dir must be empty")
    torch.set_num_threads(2)
    write_synthetic_dataset(work / "data" / "raw")
    prepare_idrid(work / "data" / "raw", work / "data" / "prepared", strict=False)
    cfg = smoke_config(work / "data", work / "results", ROOT / "configs/hybrid_idrid_fixed_iterations_no_spatial.yaml")
    cfg["split"]["val_size"] = 0.0
    cfg["data"]["train_batches_per_epoch"] = 1  # Deliberately tiny synthetic check, not the real 22-step schedule.
    payload = run(cfg, allow_synthetic=True)
    split = json.loads((work / "results" / "training_split.json").read_text())
    assert len(split["train"]) == 20 and len(split["test"]) == 10
    assert set(split["train"]).isdisjoint(split["test"])
    assert payload["optimizer_steps"] == 2 and payload["final_epoch"] == 2
    assert payload["test"]["samples"] == 10
    assert payload["official_test_used_for_training"] is False
    assert len(payload["test"]["per_class"]) == 5
    assert (work / "results" / "per_class_metrics.csv").is_file()
    print(f"PASS: EfficientNetV2-S + GeM, no spatial, both phases, fixed updates, held-out TEST and five class accuracies.\nSynthetic artifacts: {work}")


if __name__ == "__main__":
    main()
