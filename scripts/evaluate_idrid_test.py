"""Evaluate an existing final.pt on official IDRiD test; never retrain or prepare data."""
from __future__ import annotations

import argparse
import copy
import json
import shlex
from pathlib import Path

import torch

import train_idrid as hybrid
from eyepacs_hybrid_ordinal.models import HybridOrdinalNet, clean_state_dict_keys
from eyepacs_hybrid_ordinal.utils import resolve_device, save_json


def read_run_json(path):
    """Read optional status metadata without loading a large checkpoint."""
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def final_checkpoint_path(run_dir):
    """Explain an incorrect folder or unfinished run before calling torch.load."""
    hint = "List existing runs with: python scripts/evaluate_idrid_test.py --list-runs"
    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"Run directory does not exist: {run_dir}\n"
            f"--run-dir must be the ACTUAL training output folder, not an example name. {hint}"
        )
    path = run_dir / "final.pt"
    if path.is_file():
        return path
    message = f"final.pt is missing in {run_dir}. "
    if (run_dir / "last.pt").is_file():
        saved = read_run_json(run_dir / "run_config.json")
        epoch = read_run_json(run_dir / "metrics.json").get("epoch", "unknown")
        planned = saved.get("config", {}).get("train", {}).get("epochs", "unknown")
        message += (
            f"last.pt exists (recorded epoch {epoch}/{planned}); training may be unfinished or interrupted. "
            "final.pt is written only after the final epoch; do not rename an intermediate last.pt. "
        )
    elif (run_dir / "best.pt").is_file():
        message += "This folder has a validation-selected best.pt from a different protocol; this runner requires a train-only final.pt. "
    else:
        message += "Verify the actual output folder and training completion. "
    raise FileNotFoundError(message + hint)


def list_runs(search_root):
    """List existing IDRiD folders and print exact commands for final checkpoints."""
    search_root = search_root.resolve()
    if not search_root.is_dir():
        print(f"Checkpoint directory does not exist: {search_root}")
        return []
    folders = set()
    for name in ("run_config.json", "final.pt", "last.pt", "best.pt"):
        folders.update(path.parent for path in search_root.rglob(name))
    records = []
    for folder in sorted(folders):
        saved = read_run_json(folder / "run_config.json")
        if saved.get("training_performed") is False:
            continue
        cfg = saved.get("config", {})
        data_dir = str(cfg.get("data", {}).get("data_dir", ""))
        relative_name = str(folder.relative_to(search_root))
        if "idrid" not in relative_name.lower() and "idrid" not in data_dir.lower():
            continue
        seen = saved.get("official_test_used_for_training") is True
        if seen:
            status = "TRAIN-SEEN: not eligible for held-out analysis"
        elif (folder / "final.pt").is_file():
            status = "FINAL FOUND: checkpoint protocol will be verified on evaluation"
        elif (folder / "last.pt").is_file():
            epoch = read_run_json(folder / "metrics.json").get("epoch", "unknown")
            planned = cfg.get("train", {}).get("epochs", "unknown")
            status = f"NO FINAL: last checkpoint at {epoch}/{planned}; unfinished/interrupted"
        elif (folder / "best.pt").is_file():
            status = "BEST ONLY: different validation-selected protocol"
        else:
            status = "NO CHECKPOINT: setup only or training has not saved a model"
        print(f"\n{folder}\n  {status}")
        record = {"run_dir": str(folder), "status": status}
        if not seen and (folder / "final.pt").is_file():
            command = f"python scripts/evaluate_idrid_test.py --run-dir {shlex.quote(str(folder))} --device cuda"
            record["command"] = command
            print(f"  {command}")
        records.append(record)
    if not records:
        print(f"No IDRiD runs found under {search_root}; use --search-root /actual/checkpoints/path if needed.")
    return records


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
    checkpoint_path = final_checkpoint_path(run_dir)
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
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--run-dir", type=Path, help="ACTUAL training output folder containing final.pt and training metadata")
    action.add_argument("--list-runs", action="store_true", help="List existing run folders, checkpoint status and evaluation commands; no training/evaluation")
    parser.add_argument("--search-root", type=Path, help="Checkpoint tree for --list-runs; default: this repository's checkpoints/")
    parser.add_argument("--data-dir", type=Path, help="Override prepared-data location if needed")
    parser.add_argument("--output-dir", type=Path, help="Default: RUN_DIR/test_analysis; must be empty")
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    args = parser.parse_args()
    if args.list_runs:
        list_runs(args.search_root or hybrid.ROOT / "checkpoints")
        return
    if args.search_root is not None:
        parser.error("--search-root is only used with --list-runs.")
    options = vars(args)
    options.pop("list_runs")
    options.pop("search_root")
    try:
        run(**options)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"IDRiD ANALYSIS ERROR: {error}\n")


if __name__ == "__main__":
    main()
