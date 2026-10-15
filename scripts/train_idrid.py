"""Select a training plan on IDRiD train only, refit all 413, then evaluate 103."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import shlex
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eyepacs_hybrid_ordinal.data import ClassStratifiedBatchSampler, EyePACSDataset
from eyepacs_hybrid_ordinal.idrid import OFFICIAL_COUNTS, audit_duplicates, digest
from eyepacs_hybrid_ordinal.losses import PrototypeContrastiveOrdinalLoss, WeightedSupervisedContrastiveOrdinalLoss
from eyepacs_hybrid_ordinal.metrics import aggregate_predictions
from eyepacs_hybrid_ordinal.models import HybridOrdinalNet
from eyepacs_hybrid_ordinal.transforms import make_idrid_transform
from eyepacs_hybrid_ordinal.utils import build_optimizer, load_config, resolve_device, save_checkpoint, save_json, set_seed, worker_count
from train_hybrid_ordinal import build_learnable_margins, build_scheduler, evaluate, train_one_epoch


def load_prepared(data_dir: Path, *, allow_synthetic: bool = False):
    prepared = data_dir / "prepared"
    manifest_path = prepared / "manifest.json"
    if not manifest_path.is_file():
        command = shlex.join(["python", "scripts/prepare_idrid.py", "--data-dir", str(data_dir)])
        raise FileNotFoundError(
            f"IDRiD preparation has not completed: {manifest_path} is missing. "
            f"Run `{command}` successfully before training. If preparation reports duplicates, "
            f"inspect {prepared / 'duplicate_audit.json'}; --overlap-policy report preserves "
            "the supplied split with disclosed overlap and separate non-overlap test metrics."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest["official_split_verified"] and not allow_synthetic:
        raise ValueError("An official 413/103 split is required. Run scripts/prepare_idrid.py first.")
    frames = {}
    for split in ("train", "test"):
        csv_path = prepared / f"{split}.csv"
        expected_hash = manifest.get("prepared_csv_sha256", {}).get(split)
        if expected_hash is not None and digest(csv_path) != expected_hash:
            raise ValueError(f"Prepared {split}.csv has changed; rerun scripts/prepare_idrid.py.")
        frame = pd.read_csv(csv_path)
        labels = pd.to_numeric(frame["level"], errors="raise")
        if not labels.isin(range(5)).all():
            raise ValueError(f"Invalid labels in {split}.csv")
        counts = [int((labels == k).sum()) for k in range(5)]
        if not allow_synthetic and counts != OFFICIAL_COUNTS[split]:
            raise ValueError(f"Unexpected official {split} counts: {counts}")
        if frame["sample_id"].duplicated().any() or frame["image"].duplicated().any():
            raise ValueError(f"Duplicate samples in {split}.csv")
        frame["_image_id"] = frame["image"]
        frame["_label"] = labels.astype(int)
        frames[split] = frame
    for column in ("image", "sample_id"):
        if set(frames["train"][column]) & set(frames["test"][column]):
            raise ValueError(f"Train/test overlap in {column}.")
    audit = audit_duplicates({split: frame.to_dict("records") for split, frame in frames.items()})
    overlap_count = audit["summary"]["overlapping_test_samples"]
    if overlap_count:
        if manifest.get("overlap_policy", "error") != "report":
            raise ValueError("Train/test image-content overlap; rerun prepare_idrid.py --overlap-policy report to disclose and preserve it.")
        warnings.warn(
            f"{overlap_count} test images also occur in training. Full-test metrics include "
            "these duplicates; train-overlap-excluded metrics will also be reported.", UserWarning,
        )
    manifest["duplicate_summary"] = audit["summary"]
    train_hashes = set(frames["train"]["sha256"])
    frames["test"]["_overlaps_train"] = frames["test"]["sha256"].isin(train_hashes)
    return frames, manifest


def make_loader(frame, manifest, cfg, *, training):
    data = cfg["data"]
    dataset = EyePACSDataset(
        frame,
        image_root=manifest["image_root"],
        transform=make_idrid_transform(
            training=training, normalize=data["normalize"], augmentation=data.get("augmentation"),
        ),
        verify_images=True,
    )
    workers = worker_count(int(data["num_workers"]))
    kwargs = dict(num_workers=workers, pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)
    if training:
        sampler = ClassStratifiedBatchSampler(
            dataset.targets, batch_size=int(data["batch_size"]), seed=int(cfg["seed"]),
            min_samples_per_class=2,
        )
        return DataLoader(dataset, batch_sampler=sampler, **kwargs)
    return DataLoader(dataset, batch_size=int(data["batch_size"]), shuffle=False, **kwargs)


def build_components(cfg, device):
    model_cfg = {k: v for k, v in cfg["model"].items() if k != "num_classes"}
    model = HybridOrdinalNet(**model_cfg).to(device)
    margins = build_learnable_margins(cfg, device)
    options = cfg["loss"]
    shared = dict(
        num_classes=5, temperature=float(options["temperature"]),
        margin_scale=float(options["ordinal_margin_scale"]),
        normalize_ordinal_distance=bool(options["normalize_ordinal_distance"]),
        square_ordinal_distance=bool(options["square_ordinal_distance"]),
        reduction=options["reduction"],
    )
    criteria = {
        "pcol": PrototypeContrastiveOrdinalLoss(**shared),
        "scol": WeightedSupervisedContrastiveOrdinalLoss(**shared, objective=options["scol_objective"]),
        "mmnp": WeightedSupervisedContrastiveOrdinalLoss(
            num_classes=5, temperature=1.0, margin_scale=float(options["ordinal_margin_scale"]),
            square_ordinal_distance=False, reduction=options["mmnp_reduction"], objective="mmnp",
        ),
    }
    parameters = list(model.parameters())
    optimizer = build_optimizer(parameters, cfg)
    if margins is not None:
        parameters += list(margins.parameters())
        optimizer.add_param_group({"params": list(margins.parameters()), "lr": float(options["margin_lr"]), "weight_decay": 0.0})
    amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    return model, margins, criteria, parameters, optimizer, GradScaler(enabled=amp), amp


def margins_record(margins):
    if margins is None:
        return {"enabled": False}
    return {
        "enabled": True, "frozen": not margins.raw_margins.requires_grad,
        "values": margins.margin_values().detach().cpu().tolist(),
    }


def selection_indices(frame, *, val_size, seed):
    """Keep exact image copies together when choosing the train-only epoch plan."""
    if not frame["sha256"].duplicated().any():
        return (*train_test_split(
            np.arange(len(frame)), test_size=val_size, random_state=seed, stratify=frame["_label"],
        ), "image_stratified")
    groups = frame.groupby("sha256")["_label"].agg(["first", "nunique"])
    all_classes = set(frame["_label"])

    def valid_partition(train_idx, val_idx):
        train_counts = frame.iloc[train_idx]["_label"].value_counts()
        return (
            set(train_counts.index) == all_classes and train_counts.min() >= 2
            and set(frame.iloc[val_idx]["_label"]) == all_classes
        )

    # With consistent group labels, stratify the unique images themselves.
    if (groups["nunique"] == 1).all():
        try:
            train_hashes, val_hashes = train_test_split(
                groups.index, test_size=val_size, random_state=seed, stratify=groups["first"],
            )
            train_idx = np.flatnonzero(frame["sha256"].isin(train_hashes))
            val_idx = np.flatnonzero(frame["sha256"].isin(val_hashes))
            if valid_partition(train_idx, val_idx):
                return train_idx, val_idx, "sha256_grouped_stratified"
        except ValueError:
            pass
    # Mixed-label copies cannot be assigned a single stratification label.
    splitter = GroupShuffleSplit(n_splits=128, test_size=val_size, random_state=seed)
    candidates = []
    for train_idx, val_idx in splitter.split(frame, frame["_label"], groups=frame["sha256"]):
        if valid_partition(train_idx, val_idx):
            candidates.append((train_idx, val_idx))
    if not candidates:
        raise ValueError("Cannot construct a hash-grouped internal split with all grades; inspect duplicate_audit.json.")
    train_idx, val_idx = min(candidates, key=lambda pair: abs(len(pair[1]) / len(frame) - val_size))
    return train_idx, val_idx, "sha256_grouped_class_coverage"


def select_plan(frame, manifest, cfg, output, device):
    """Never receives the official test frame or uses its labels for selection."""
    set_seed(int(cfg["seed"]))
    train_idx, val_idx, split_method = selection_indices(
        frame, val_size=float(cfg["split"]["val_size"]), seed=int(cfg["seed"]),
    )
    training, validation = frame.iloc[train_idx].copy(), frame.iloc[val_idx].copy()
    if set(training["sha256"]) & set(validation["sha256"]):
        raise ValueError("Duplicate image contents cross the internal train/validation split.")
    save_json(output / "selection_split.json", {
        "train": training["sample_id"].tolist(), "validation": validation["sample_id"].tolist(),
        "method": split_method, "patient_grouping": "Unavailable in the public grading CSV; content groups are not patient IDs.",
    })
    train_loader = make_loader(training, manifest, cfg, training=True)
    val_loader = make_loader(validation, manifest, cfg, training=False)
    model, margins, criteria, parameters, optimizer, scaler, amp = build_components(cfg, device)
    scheduler = build_scheduler(optimizer, cfg)
    phase = "phase1_joint" if margins is not None else "fixed"
    previous = None
    stable_epochs, bad_epochs, best_epoch = 0, 0, 0
    best_key = (float("-inf"), float("-inf"))
    best_val = None
    history = []
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        learning_rates = [float(g["lr"]) for g in optimizer.param_groups]
        train_metrics = train_one_epoch(
            model, train_loader, train_loader, optimizer, scaler, device, amp,
            criteria, margins, parameters, cfg, epoch,
        )
        val_metrics = evaluate(model, val_loader, device, amp, cfg)
        if val_metrics.get("nonfinite_predictions", 0):
            raise FloatingPointError("Non-finite validation predictions.")
        record = {
            "epoch": epoch, "phase": phase, "learning_rates": learning_rates,
            "train": train_metrics, "val": val_metrics, "margins": margins_record(margins),
        }
        history.append(record)
        # Only phase-2 models are eligible when learnable margins are enabled.
        if phase != "phase1_joint":
            key = (float(val_metrics["accuracy"]), -float(val_metrics["mae"]))
            if key > best_key:
                best_key, best_epoch, best_val, bad_epochs = key, epoch, val_metrics, 0
                save_checkpoint(output / "selection_best.pt", {
                    "model": model.state_dict(), "epoch": epoch,
                    "margins": margins.state_dict() if margins is not None else None,
                })
            else:
                bad_epochs += 1
        scheduler.step(float(val_metrics["mae"]))
        print(f"selection epoch={epoch} phase={phase} val_accuracy={val_metrics['accuracy']:.4f} val_mae={val_metrics['mae']:.4f}")
        save_json(output / "selection_history.json", {"epochs": history})
        if phase == "phase1_joint":
            options = cfg["loss"]
            current = margins.margin_values().detach().clone()
            change = float((current - previous).abs().max()) if previous is not None else float("inf")
            stable_epochs = stable_epochs + 1 if change <= float(options["margin_convergence_tolerance"]) else 0
            previous = current
            converged = epoch >= int(options["margin_phase1_min_epochs"]) and stable_epochs >= int(options["margin_convergence_patience"])
            if converged or epoch >= int(options["margin_phase1_max_epochs"]):
                margins.freeze()
                phase = "phase2_frozen"
                optimizer.param_groups[0]["lr"] = float(cfg["train"]["lr"])
                optimizer.param_groups[1]["lr"] = float(options["margin_lr"])
                scheduler = build_scheduler(optimizer, cfg)
                bad_epochs = 0
        elif bad_epochs >= int(cfg["train"]["early_stopping_patience"]):
            break
    if not best_epoch:
        raise RuntimeError("No eligible checkpoint was selected; allow at least one phase-2 epoch.")
    plan = {
        "selection_metric": "internal_val.accuracy (higher); ties: internal_val.mae (lower)",
        "best_epoch": best_epoch, "validation": best_val,
        "epochs": [{k: row[k] for k in ("epoch", "phase", "learning_rates")} for row in history[:best_epoch]],
    }
    save_json(output / "training_plan.json", plan)
    return plan


def refit(frame, manifest, cfg, output, device, plan):
    """Fresh model, full official train split, fixed epoch/freeze/LR plan; no test access."""
    set_seed(int(cfg["seed"]))
    train_loader = make_loader(frame, manifest, cfg, training=True)
    model, margins, criteria, parameters, optimizer, scaler, amp = build_components(cfg, device)
    history = []
    for step in plan["epochs"]:
        if margins is not None and step["phase"] == "phase2_frozen":
            margins.freeze()
        for group, lr in zip(optimizer.param_groups, step["learning_rates"]):
            group["lr"] = lr
        train_metrics = train_one_epoch(
            model, train_loader, train_loader, optimizer, scaler, device, amp,
            criteria, margins, parameters, cfg, step["epoch"],
        )
        history.append({**step, "train": train_metrics, "margins": margins_record(margins)})
        save_json(output / "metrics_history.json", {"epochs": history})
        print(f"refit epoch={step['epoch']} phase={step['phase']} train_accuracy={train_metrics['accuracy']:.4f}")
    save_checkpoint(output / "best.pt", {
        "model": model.state_dict(), "epoch": plan["best_epoch"], "config": cfg,
        "margins": margins.state_dict() if margins is not None else None,
        "selection": "Final full-training refit of the internal-validation-selected epoch plan.",
    })
    return model, amp


def test_metrics(predictions, labels):
    if not len(labels):
        return None
    metrics = aggregate_predictions(list(predictions), list(labels), 5)
    classes = np.rint(predictions).clip(0, 4).astype(int)
    confusion = np.zeros((5, 5), dtype=int)
    np.add.at(confusion, (np.asarray(labels), classes), 1)
    metrics.update(samples=len(labels), accuracy_percent=100 * metrics["accuracy"], confusion_matrix=confusion.tolist())
    return metrics


@torch.no_grad()
def evaluate_test(model, frame, manifest, cfg, output, device, amp):
    loader = make_loader(frame, manifest, cfg, training=False)
    model.eval()
    predictions, labels, ids = [], [], []
    for images, targets, image_ids in loader:
        with autocast(enabled=amp):
            batch = model(images.to(device))["prediction"].float().cpu()
        if not torch.isfinite(batch).all():
            raise FloatingPointError("Non-finite test predictions; no metrics will be reported.")
        predictions.extend(batch.tolist())
        labels.extend(targets.tolist())
        ids.extend(image_ids)
    metrics = test_metrics(predictions, labels)
    classes = np.rint(predictions).clip(0, 4).astype(int)
    overlap = frame.set_index("_image_id")["_overlaps_train"].loc[ids].to_numpy(dtype=bool)
    filtered_metrics = test_metrics(np.asarray(predictions)[~overlap], np.asarray(labels)[~overlap])
    payload = {
        "dataset": "IDRiD", "split": "official_test", "checkpoint": "best.pt",
        "synthetic_smoke_test": not manifest["official_split_verified"],
        "accuracy_definition": "correct discrete grade predictions / N (micro accuracy)",
        "mae_definition": "mean(abs(round_and_clip(prediction, 0, 4) - true_grade))",
        "test": metrics,
        "data_integrity": {
            "overlap_policy": manifest.get("overlap_policy", "error"),
            "duplicate_summary": manifest.get("duplicate_summary", {}),
            "full_test_contains_train_duplicates": bool(overlap.any()),
        },
        "test_without_train_overlap": {
            "excluded_samples": int(overlap.sum()), "samples": int((~overlap).sum()),
            "filter": "Exclude test images whose file SHA-256 occurs in official training; other similarities are not checked.",
            "metrics": filtered_metrics,
            "status": "available" if filtered_metrics is not None else "unavailable_all_test_images_overlap",
        },
    }
    save_json(output / "evaluation_metrics.json", payload)
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "label", "prediction_continuous", "prediction_class", "absolute_error", "overlaps_train"])
        writer.writerows((i, y, p, int(c), abs(int(c) - y), bool(o)) for i, y, p, c, o in zip(ids, labels, predictions, classes, overlap))
    print(f"IDRiD full test: Accuracy={metrics['accuracy_percent']:.2f}% MAE={metrics['mae']:.4f} N={len(labels)} train_overlaps={int(overlap.sum())}")
    if filtered_metrics is not None:
        print(f"Test excluding train overlaps: Accuracy={filtered_metrics['accuracy_percent']:.2f}% MAE={filtered_metrics['mae']:.4f} N={filtered_metrics['samples']}")
    else:
        print("Test excluding train overlaps: unavailable (all test images overlap training).")
    return payload


def run(cfg, *, allow_synthetic=False):
    if int(cfg["data"]["image_size"]) != 224 or int(cfg["model"]["num_classes"]) != 5:
        raise ValueError("This IDRiD protocol requires 224x224 input and five DR grades.")
    output = Path(cfg["train"]["output_dir"]).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty; choose a new --output-dir: {output}")
    frames, manifest = load_prepared(Path(cfg["data"]["data_dir"]), allow_synthetic=allow_synthetic)
    device = resolve_device(cfg["device"])
    output.mkdir(parents=True, exist_ok=True)
    git_rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    save_json(output / "run_config.json", {
        "config": cfg, "dataset": manifest, "git_commit": git_rev.stdout.strip(),
        "python": platform.python_version(), "torch": str(torch.__version__),
        "device": str(device), "synthetic_smoke_test": allow_synthetic,
        "protocol": "internal train-only selection -> fresh full-train refit -> one official test evaluation",
    })
    plan = select_plan(frames["train"], manifest, cfg, output, device)
    model, amp = refit(frames["train"], manifest, cfg, output, device, plan)
    # Validate serialization/reloading before the only official test evaluation.
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    return evaluate_test(model, frames["test"], manifest, cfg, output, device, amp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hybrid_idrid_efficientnet_v2_s.yaml")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.data_dir is not None:
        cfg["data"]["data_dir"] = str(args.data_dir.resolve())
    if args.output_dir is not None:
        cfg["train"]["output_dir"] = str(args.output_dir.resolve())
    if args.device is not None:
        cfg["device"] = args.device
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    run(cfg)


if __name__ == "__main__":
    main()
