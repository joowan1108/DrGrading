"""Evaluate an existing final.pt on official IDRiD test; never retrain or prepare data."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

import train_idrid as hybrid
from eyepacs_hybrid_ordinal.models import HybridOrdinalNet, clean_state_dict_keys
from eyepacs_hybrid_ordinal.utils import resolve_device, save_json


def build_evaluation_model(cfg, device):
    options = {k: v for k, v in cfg["model"].items() if k not in {"num_classes", "balanced_softmax"}}
    options["pretrained_imagenet"] = False  # All weights come from the existing checkpoint.
    if cfg["model"].get("balanced_softmax", False):
        from eyepacs_hybrid_ordinal.balanced_softmax import BalancedSoftmaxHybridOrdinalNet
        model = BalancedSoftmaxHybridOrdinalNet(num_classes=int(cfg["model"]["num_classes"]), **options)
    else:
        model = HybridOrdinalNet(**options)
    return model.to(device)


def run(run_dir: Path, *, data_dir=None, output_dir=None, device=None, num_workers=None, allow_synthetic=False):
    run_dir = run_dir.resolve()
    checkpoint_path = run_dir / "final.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("official_test_used_for_training") is not False:
        raise ValueError("This checkpoint does not establish train-only training; a TRAIN-SEEN checkpoint cannot be relabelled as held-out test.")
    if "heldout_test" not in checkpoint.get("protocol", ""):
        raise ValueError("Expected the existing train-only/final-epoch IDRiD protocol.")
    split = json.loads((run_dir / "training_split.json").read_text(encoding="utf-8"))
    saved_run = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    cfg = copy.deepcopy(checkpoint["config"])
    cfg.setdefault("evaluation", {}).update(tsne_correct_only=True)
    if data_dir is not None:
        cfg["data"]["data_dir"] = str(Path(data_dir).resolve())
    if device is not None:
        cfg["device"] = device
    if num_workers is not None:
        cfg["data"]["num_workers"] = num_workers
    output = Path(output_dir).resolve() if output_dir is not None else run_dir / "test_analysis"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Analysis directory is not empty; choose a new --output-dir: {output}")
    frames, manifest = hybrid.load_prepared(Path(cfg["data"]["data_dir"]), allow_synthetic=allow_synthetic)
    test = frames["test"].copy()
    training_ids = split["train"]
    if not training_ids or len(set(training_ids)) != len(training_ids) or not all(s.startswith("train:") for s in training_ids):
        raise ValueError("Recorded training membership is invalid or contains official test IDs.")
    if len(split["test"]) != len(test) or set(split["test"]) != set(test["sample_id"]):
        raise ValueError("Current test membership differs from the saved training run.")
    official_train = frames["train"].set_index("sample_id", drop=False)
    if not set(training_ids) <= set(official_train.index):
        raise ValueError("Recorded training samples are missing from current prepared data.")
    training = official_train.loc[training_ids].copy()
    if "training_pixel_hashes" in split and set(split["training_pixel_hashes"]) != set(training["_pixel_sha256"]):
        raise ValueError("Training pixels differ from the original run.")
    if "test_pixel_hashes" in split and split["test_pixel_hashes"] != test["_pixel_sha256"].tolist():
        raise ValueError("Test pixels differ from the original run.")
    test["_overlaps_train"] = (
        test["sha256"].isin(set(training["sha256"]))
        | test["_pixel_sha256"].isin(set(training["_pixel_sha256"]))
    )
    if test["_overlaps_train"].any():
        raise ValueError("This saved model trained on a copy of test pixels; refusing held-out analysis.")
    resolved_device = resolve_device(cfg["device"])
    model = build_evaluation_model(cfg, resolved_device)
    model.load_state_dict(clean_state_dict_keys(checkpoint["model"]), strict=True)
    output.mkdir(parents=True, exist_ok=True)
    # Keep the original checkpoint/membership; only add pixel audit fields for older runs.
    analysis_split = {**split, "training_pixel_hashes": sorted(set(training["_pixel_sha256"])),
                      "test_pixel_hashes": test["_pixel_sha256"].tolist()}
    save_json(output / "training_split.json", analysis_split)
    save_json(output / "run_config.json", {
        "config": cfg, "dataset": manifest, "source_run": str(run_dir),
        "source_checkpoint": str(checkpoint_path), "training_git_commit": saved_run.get("git_commit"),
        "training_performed": False,
    })
    payload = hybrid.evaluate_test(
        model, test, manifest, cfg, output, resolved_device,
        bool(cfg["train"].get("amp", False)) and resolved_device.type == "cuda",
        checkpoint_name="final.pt", training_frame=training, require_heldout=True,
    )
    payload.update({
        "source_run": str(run_dir), "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]), "protocol": checkpoint["protocol"],
        "official_test_used_for_training": False, "evaluation_is_independent": True,
        "training_performed": False,
        "independence_scope": "Same saved train-only run and test membership; does not establish patient independence or undo prior test use.",
    })
    save_json(output / "evaluation_metrics.json", payload)
    print(f"Test analysis saved to {output}; no training or data preparation was run.", flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="Existing IDRiD run with final.pt, run_config.json and training_split.json")
    parser.add_argument("--data-dir", type=Path, help="Override prepared-data location if needed")
    parser.add_argument("--output-dir", type=Path, help="Default: RUN_DIR/test_analysis; must be empty")
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
