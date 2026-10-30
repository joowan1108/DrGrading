"""Train on official IDRiD TRAIN only, preserving 22 updates/epoch; evaluate TEST once.

No internal validation, no test-driven scheduler/checkpoint selection. Exact image
copies of test files are removed from training, never from the evaluation set.
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import subprocess
from pathlib import Path

import torch

import train_idrid as hybrid
from train_idrid_all_data import fit_all
from eyepacs_hybrid_ordinal.utils import load_config, resolve_device, save_json

PROTOCOL = "official_train_only_no_validation_fixed_iterations_final_epoch_heldout_test"


def exclude_test_copies(training, test):
    """Keep every official test row; exclude byte-identical training copies."""
    overlap = training["sha256"].isin(set(test["sha256"]))
    excluded = training.loc[overlap].copy()
    clean = training.loc[~overlap].copy()
    counts = clean["_label"].value_counts()
    if set(counts.index) != set(range(5)) or int(counts.min()) < 2:
        raise ValueError("Removing test copies leaves fewer than two training images in a grade; inspect duplicate_audit.json.")
    if set(clean["sha256"]) & set(test["sha256"]):
        raise ValueError("Test image contents still overlap training.")
    return clean, excluded


def run(cfg, *, allow_synthetic=False):
    cfg = copy.deepcopy(cfg)
    if int(cfg["data"]["image_size"]) != 224 or int(cfg["model"]["num_classes"]) != 5:
        raise ValueError("This IDRiD protocol requires 224x224 input and five DR grades.")
    if float(cfg.get("split", {}).get("val_size", 0.0)) != 0.0:
        raise ValueError("Use val_size: 0.0: this script has NO internal validation.")
    batches = cfg["data"].get("train_batches_per_epoch")
    if isinstance(batches, bool) or not isinstance(batches, int) or batches < 1:
        raise ValueError("Set train_batches_per_epoch to a positive integer (default 22).")
    epochs = int(cfg["train"]["epochs"])
    if epochs < 1:
        raise ValueError("epochs must be positive.")
    if cfg["loss"].get("learnable_ordinal_margins", False):
        minimum = int(cfg["loss"]["margin_phase1_min_epochs"])
        maximum = int(cfg["loss"]["margin_phase1_max_epochs"])
        if not 1 <= minimum <= maximum < epochs:
            raise ValueError("Require 1 <= phase1_min <= phase1_max < epochs so Phase 2 can run.")
    cfg["split"] = {"val_size": 0.0}
    cfg["train"].update(scheduler_metric="train.rmse", early_stopping_patience=None, checkpoint_selection="final_epoch")
    output = Path(cfg["train"]["output_dir"]).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty; choose a new --output-dir: {output}")
    frames, manifest = hybrid.load_prepared(Path(cfg["data"]["data_dir"]), allow_synthetic=allow_synthetic)
    training, excluded = exclude_test_copies(frames["train"], frames["test"])
    test = frames["test"].copy()
    # Refresh membership against the ACTUAL pool, not the pre-cleaning official train CSV.
    test["_overlaps_train"] = test["sha256"].isin(set(training["sha256"]))
    device = resolve_device(cfg["device"])
    output.mkdir(parents=True, exist_ok=True)
    run_metadata = {
        "protocol": PROTOCOL, "official_test_used_for_training": False,
        "evaluation_is_independent": True,
        "independence_scope": "Held out from this fresh run; file-hash disjointness does not establish patient independence or undo prior test-set use.",
        "batches_per_epoch": batches, "planned_optimizer_steps": epochs * batches,
    }
    split = {
        "train": training["sample_id"].tolist(), "validation": [],
        "test": test["sample_id"].tolist(), "independent_test": test["sample_id"].tolist(),
        "official_test_used_for_training": [],
        "source_counts": {name: len(frame) for name, frame in frames.items()},
        "training_samples": len(training), "test_samples": len(test),
        "excluded_training_test_copies": excluded["sample_id"].tolist(),
        "unique_training_hashes": int(training["sha256"].nunique()),
        "class_counts": [int((training["_label"] == grade).sum()) for grade in range(5)],
        "method": "official_train_only_exclude_exact_test_copies_no_validation",
        "patient_grouping": "Patient identifiers are unavailable; SHA-256 only detects identical files.",
    }
    save_json(output / "training_split.json", split)
    git_rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=hybrid.ROOT, capture_output=True, text=True)
    save_json(output / "run_config.json", {
        "config": cfg, "dataset": manifest, "training_split": split,
        "git_commit": git_rev.stdout.strip(), "python": platform.python_version(),
        "torch": str(torch.__version__), "device": str(device),
        "synthetic_smoke_test": allow_synthetic, **run_metadata,
    })
    print(
        f"TRAIN ONLY: {len(training)} images; TEST held out: {len(test)} images; "
        f"training copies of test removed: {len(excluded)}.\n"
        f"Fixed optimizer budget: {batches} updates/epoch x {epochs} epochs = {batches * epochs} updates.\n"
        f"Model={cfg['model']['backbone']} pooling={cfg['model']['pooling']} "
        f"spatial_attention={cfg['model'].get('spatial_attention', False)}.\n"
        "Fresh initialization from the configured ImageNet weights; no old IDRiD checkpoint is loaded. "
        "No validation/early stopping/test-based selection.", flush=True,
    )
    model, amp, summary = fit_all(
        training, manifest, cfg, output, device,
        protocol=PROTOCOL, official_test_used_for_training=False,
    )
    checkpoint = torch.load(output / "final.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    payload = hybrid.evaluate_test(
        model, test, manifest, cfg, output, device, amp, checkpoint_name="final.pt",
    )
    payload.update({
        **run_metadata, **summary, "checkpoint_selection": "final_epoch",
        "excluded_training_test_copies": excluded["sample_id"].tolist(),
        "per_class_accuracy_definition": "Correct predictions within each true class / that class's support (recall).",
    })
    save_json(output / "evaluation_metrics.json", payload)
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    metrics["test"] = payload["test"]
    save_json(output / "metrics.json", metrics)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hybrid_idrid_fixed_iterations_no_spatial.yaml")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    for name, section, key in (
        ("data_dir", "data", "data_dir"), ("output_dir", "train", "output_dir"),
        ("num_workers", "data", "num_workers"), ("epochs", "train", "epochs"),
    ):
        value = getattr(args, name)
        if value is not None:
            cfg[section][key] = str(value.resolve()) if isinstance(value, Path) else value
    if args.device is not None:
        cfg["device"] = args.device
    run(cfg)


if __name__ == "__main__":
    main()
