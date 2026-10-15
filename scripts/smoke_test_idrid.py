"""Offline end-to-end test with synthetic images; never downloads IDRiD/weights."""
from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from train_idrid import ROOT, run
from eyepacs_hybrid_ordinal.idrid import prepare_idrid
from eyepacs_hybrid_ordinal.utils import load_config


def write_synthetic_dataset(root: Path, counts=None):
    counts = counts or {"train": [4] * 5, "test": [2] * 5}
    rng = np.random.default_rng(123)
    grading = root / "B.%20Disease%20Grading" / "B. Disease Grading"
    for split, label_counts in counts.items():
        name = "Training" if split == "train" else "Testing"
        prefix = "a" if split == "train" else "b"
        images = grading / "1. Original Images" / f"{prefix}. {name} Set"
        labels = grading / "2. Groundtruths" / f"{prefix}. IDRiD_Disease Grading_{name} Labels.csv"
        images.mkdir(parents=True, exist_ok=True)
        labels.parent.mkdir(parents=True, exist_ok=True)
        with labels.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Image name", "Retinopathy grade", "Risk of macular edema "])
            index = 0
            for grade, count in enumerate(label_counts):
                for _ in range(count):
                    index += 1
                    stem = f"IDRiD_{index:03d}"
                    pixels = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
                    Image.fromarray(pixels).save(images / f"{stem}.jpg")
                    writer.writerow([stem, grade, 0])


def smoke_config(data_dir: Path, output_dir: Path, config_path=None):
    cfg = load_config(config_path or ROOT / "configs/hybrid_idrid_efficientnet_v2_s.yaml")
    cfg["device"] = "cpu"
    cfg["data"].update(data_dir=str(data_dir), batch_size=10, num_workers=0)
    cfg["model"].update(pretrained_imagenet=False, projection_hidden_dim=32, projection_dim=8, regression_hidden_dim=32)
    cfg["split"]["val_size"] = 0.25
    cfg["loss"].update(margin_phase1_min_epochs=1, margin_phase1_max_epochs=1)
    cfg["train"].update(epochs=2, amp=False, output_dir=str(output_dir))
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--config", type=Path, help="Configuration to smoke-test with the short synthetic schedule")
    args = parser.parse_args()
    work = args.work_dir or Path(tempfile.mkdtemp(prefix="idrid-smoke-"))
    if work.exists() and any(work.iterdir()):
        parser.error("--work-dir must be empty")
    torch.set_num_threads(2)
    source = work / "data" / "raw"
    write_synthetic_dataset(source)
    prepare_idrid(source, work / "data" / "prepared", strict=False)
    payload = run(smoke_config(work / "data", work / "results", args.config), allow_synthetic=True)
    history = json.loads((work / "results" / "metrics_history.json").read_text())
    assert [r["phase"] for r in history["epochs"]] == ["phase1_joint", "phase2_frozen"]
    assert history["epochs"][0]["margins"]["values"] == history["epochs"][1]["margins"]["values"]
    assert payload["test"]["samples"] == 10
    assert payload["synthetic_smoke_test"] is True
    assert payload["selection_metric"] == "val.rmse_loss"
    assert not (work / "results" / "training_plan.json").exists()
    print(f"PASS: existing hybrid protocol, PCOL/SCOLw weights 0.1, pair-mean MMNP, real EfficientNetV2-S, both phases, best-checkpoint test evaluation.\nSynthetic artifacts: {work}")


if __name__ == "__main__":
    main()
