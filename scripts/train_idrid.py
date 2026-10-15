"""Train IDRiD with our existing hybrid protocol, then evaluate the best model."""
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
            pair_reduction=options.get("mmnp_pair_reduction", "mean"),
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


def loss_breakdown(metrics, cfg):
    """Display each raw loss and its weighted contribution to the total."""
    weights = {
        "pcol": cfg["loss"]["alpha"], "scol": cfg["loss"]["beta"],
        "rmse": cfg["loss"]["rmse_weight"], "mmnp": cfg["loss"]["mmnp_weight"],
    }
    parts = [f"loss={metrics['total']:.4f}"]
    for name, weight in weights.items():
        parts.append(f"{name}={metrics[name]:.4f}(weighted={float(weight) * metrics[name]:.4f})")
    return " ".join(parts)


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


def fit(frame, manifest, cfg, output, device):
    """Existing train_hybrid_ordinal protocol; official test data never enters training."""
    set_seed(int(cfg["seed"]))
    train_idx, val_idx, split_method = selection_indices(
        frame, val_size=float(cfg["split"]["val_size"]), seed=int(cfg["seed"]),
    )
    training, validation = frame.iloc[train_idx].copy(), frame.iloc[val_idx].copy()
    if set(training["sha256"]) & set(validation["sha256"]):
        raise ValueError("Duplicate image contents cross the internal train/validation split.")
    save_json(output / "training_split.json", {
        "train": training["sample_id"].tolist(), "validation": validation["sample_id"].tolist(),
        "method": split_method, "patient_grouping": "Unavailable in the public grading CSV; content groups are not patient IDs.",
    })
    train_loader = make_loader(training, manifest, cfg, training=True)
    val_loader = make_loader(validation, manifest, cfg, training=False)
    model, margins, criteria, parameters, optimizer, scaler, amp = build_components(cfg, device)
    initial_margins = margins_record(margins)
    options = cfg["loss"]
    initial_margins["initialization"] = {
        "method": "uniform" if options.get("margin_init_min") is not None else "fixed",
        "uniform_range": [options.get("margin_init_min"), options.get("margin_init_max")],
        "seed": int(cfg["seed"]), "margin_lr": float(options["margin_lr"]),
    }
    save_json(output / "initial_margins.json", initial_margins)
    if margins is not None:
        print(
            f"Initial margins ({initial_margins['initialization']['method']}): "
            + ", ".join(f"{value:.6f}" for value in initial_margins["values"])
            + f" | margin_lr={float(options['margin_lr']):.2e}",
            flush=True,
        )
    scheduler = build_scheduler(optimizer, cfg)
    phase = "phase1_joint" if margins is not None else "fixed"
    previous = None
    stable_epochs, bad_epochs, best_epoch = 0, 0, 0
    best_val_rmse = float("inf")
    best_val, best_train, best_margins = {}, {}, {}
    freeze_reason = None
    history = []
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        learning_rates = [float(g["lr"]) for g in optimizer.param_groups]
        phase_label = {
            "phase1_joint": "Phase 1: joint", "phase2_frozen": "Phase 2: frozen", "fixed": "Fixed margins",
        }[phase]
        margin_status = "trainable" if margins is not None and margins.raw_margins.requires_grad else "fixed"
        print(
            f"[{phase_label}] Epoch {epoch}/{cfg['train']['epochs']} | "
            f"network_lr={learning_rates[0]:.2e} | margins={margin_status}",
            flush=True,
        )
        train_metrics = train_one_epoch(
            model, train_loader, train_loader, optimizer, scaler, device, amp,
            criteria, margins, parameters, cfg, epoch, phase=phase_label,
        )
        val_metrics = evaluate(model, val_loader, device, amp, cfg)
        if val_metrics.get("nonfinite_predictions", 0):
            raise FloatingPointError("Non-finite validation predictions.")
        scheduler.step(float(val_metrics["rmse_loss"]))
        change = 0.0 if phase == "phase2_frozen" else None
        if phase == "phase1_joint":
            current = margins.margin_values().detach().clone()
            change = float((current - previous).abs().max()) if previous is not None else None
            stable_epochs = stable_epochs + 1 if change is not None and change <= float(cfg["loss"]["margin_convergence_tolerance"]) else 0
            previous = current
        current_margins = {
            **margins_record(margins), "phase": phase, "freeze_reason": freeze_reason,
            "stable_epochs": stable_epochs, "max_change": change,
        }
        if margins is not None:
            current_margins["initial_values"] = initial_margins["values"]
            current_margins["delta_from_initial"] = [
                value - initial for value, initial in zip(current_margins["values"], initial_margins["values"])
            ]
        payload = {
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "config": cfg,
            "ordinal_margins": margins.state_dict() if margins is not None else None,
            "margin_state": current_margins, "train_metrics": train_metrics, "val_metrics": val_metrics,
        }
        save_checkpoint(output / "last.pt", payload)
        improved = float(val_metrics["rmse_loss"]) < best_val_rmse
        if improved:
            best_val_rmse = float(val_metrics["rmse_loss"])
            best_epoch, best_val, best_train, best_margins = epoch, val_metrics, train_metrics, current_margins
            bad_epochs = 0
            save_checkpoint(output / "best.pt", payload)
            save_json(output / "best_metrics.json", {
                "selection_metric": "val.rmse_loss", "epoch": best_epoch, "phase": phase,
                "val_rmse_loss": best_val_rmse, "train": best_train, "val": best_val, "margins": best_margins,
            })
        else:
            bad_epochs += 1
        record = {
            "epoch": epoch, "phase": phase, "learning_rates": learning_rates,
            "train": train_metrics, "val": val_metrics, "margins": current_margins, "is_best": improved,
        }
        history.append(record)
        print(
            f"epoch={epoch} phase={phase} {loss_breakdown(train_metrics, cfg)} "
            f"val_rmse={val_metrics['rmse_loss']:.4f} val_accuracy={val_metrics['accuracy']:.4f} "
            f"val_mae={val_metrics['mae']:.4f}"
        )
        if margins is not None:
            values = ", ".join(f"{value:.6f}" for value in current_margins["values"])
            delta = ", ".join(f"{value:+.6f}" for value in current_margins["delta_from_initial"])
            print(
                f"margins=[{values}] delta_from_initial=[{delta}] "
                f"max_change={change if change is not None else 'n/a'} "
                f"stable_epochs={stable_epochs} | "
                f"convergence_tolerance={float(options['margin_convergence_tolerance']):.2e}",
                flush=True,
            )
        save_json(output / "metrics_history.json", {"selection_metric": "val.rmse_loss", "epochs": history})
        save_json(output / "metrics.json", {
            "epoch": epoch, "best_epoch": best_epoch, "best_val_rmse_loss": best_val_rmse,
            "best_train": best_train, "best_val": best_val, "best_margins": best_margins,
            "train": train_metrics, "val": val_metrics, "margins": current_margins,
        })
        if phase == "phase1_joint":
            options = cfg["loss"]
            converged = epoch >= int(options["margin_phase1_min_epochs"]) and stable_epochs >= int(options["margin_convergence_patience"])
            if converged or epoch >= int(options["margin_phase1_max_epochs"]):
                freeze_reason = "margin_converged" if converged else "max_phase1_epochs"
                save_checkpoint(output / "phase1_last.pt", payload)
                margins.freeze()
                phase = "phase2_frozen"
                optimizer.param_groups[0]["lr"] = float(cfg["train"]["lr"])
                optimizer.param_groups[1]["lr"] = float(options["margin_lr"])
                scheduler = build_scheduler(optimizer, cfg)
                # Match the existing phase transition: restart checkpoint selection.
                bad_epochs = 0
                best_val_rmse, best_epoch = float("inf"), 0
                best_val, best_train, best_margins = {}, {}, {}
                values = ", ".join(f"{value:.4f}" for value in current_margins["values"])
                print(
                    f"[Phase 1 -> Phase 2] After epoch {epoch}: {freeze_reason}. "
                    f"Frozen margins=[{values}]. Phase 2 starts at epoch {epoch + 1}; "
                    f"network_lr reset to {optimizer.param_groups[0]['lr']:.2e}; "
                    "scheduler, best-checkpoint selection and early-stopping counter reset.",
                    flush=True,
                )
        elif bad_epochs >= int(cfg["train"]["early_stopping_patience"]):
            print(f"Early stopping after {bad_epochs} epochs without validation RMSE improvement.")
            break
    if not best_epoch:
        raise RuntimeError("No eligible checkpoint was selected; allow at least one phase-2 epoch.")
    return model, amp, {"best_epoch": best_epoch, "best_val_rmse_loss": best_val_rmse, "best_val": best_val}


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
    print(
        f"Loss weights: PCOL={cfg['loss']['alpha']} SCOLw={cfg['loss']['beta']} "
        f"RMSE={cfg['loss']['rmse_weight']} MMNP={cfg['loss']['mmnp_weight']} "
        f"MMNP pair_reduction={cfg['loss'].get('mmnp_pair_reduction', 'mean')}"
    )
    print(
        f"Epoch limit={cfg['train']['epochs']} | early_stopping_patience={cfg['train']['early_stopping_patience']} | "
        f"phase1_epochs={cfg['loss']['margin_phase1_min_epochs']}..{cfg['loss']['margin_phase1_max_epochs']} | "
        f"convergence_patience={cfg['loss']['margin_convergence_patience']}",
        flush=True,
    )
    output.mkdir(parents=True, exist_ok=True)
    git_rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    save_json(output / "run_config.json", {
        "config": cfg, "dataset": manifest, "git_commit": git_rev.stdout.strip(),
        "python": platform.python_version(), "torch": str(torch.__version__),
        "device": str(device), "synthetic_smoke_test": allow_synthetic,
        "protocol": "existing hybrid training: internal validation RMSE -> best checkpoint -> official test; no full-train refit",
    })
    model, amp, summary = fit(frames["train"], manifest, cfg, output, device)
    # Validate serialization/reloading before the only official test evaluation.
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    payload = evaluate_test(model, frames["test"], manifest, cfg, output, device, amp)
    payload.update(summary, selection_metric="val.rmse_loss")
    save_json(output / "evaluation_metrics.json", payload)
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    metrics["test"] = payload["test"]
    save_json(output / "metrics.json", metrics)
    return payload


def main(default_config="configs/hybrid_idrid_efficientnet_v2_s.yaml"):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default=default_config, help="Training YAML configuration")
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
