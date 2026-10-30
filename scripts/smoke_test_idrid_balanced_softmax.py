"""Offline real-model check: BS auxiliary loss, both phases, held-out test metrics."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch

from smoke_test_idrid import ROOT, smoke_config, write_synthetic_dataset
from eyepacs_hybrid_ordinal.idrid import prepare_idrid
from train_idrid_balanced_softmax import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()
    work = args.work_dir or Path(tempfile.mkdtemp(prefix="idrid-bs-smoke-"))
    if work.exists() and any(work.iterdir()):
        parser.error("--work-dir must be empty")
    torch.set_num_threads(2)
    write_synthetic_dataset(work / "data" / "raw")
    prepare_idrid(work / "data" / "raw", work / "data" / "prepared", strict=False)
    cfg = smoke_config(work / "data", work / "results", ROOT / "configs/hybrid_idrid_balanced_softmax.yaml")
    cfg["split"]["val_size"] = 0.0
    cfg["data"]["train_batches_per_epoch"] = 1
    payload = run(cfg, allow_synthetic=True)
    rows = json.loads((work / "results" / "metrics_history.json").read_text())["epochs"]
    assert [row["phase"] for row in rows] == ["phase1_joint", "phase2_frozen"]
    assert all(row["train"]["classification"] > 0 for row in rows)
    assert payload["balanced_softmax"]["class_counts"] == [4] * 5
    assert payload["test"]["samples"] == 10 and payload["optimizer_steps"] == 2
    assert payload["official_test_used_for_training"] is False
    print(f"PASS: real EfficientNetV2-S + GeM + Balanced Softmax, both phases, final checkpoint reload, held-out test and five class accuracies.\nSynthetic artifacts: {work}")


if __name__ == "__main__":
    main()
