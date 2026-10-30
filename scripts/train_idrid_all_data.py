"""Train on official IDRiD train + test, then re-evaluate the SAME test images.

No internal validation, no independent test, no performance-based model selection.
The final reported accuracy/MAE are training-data resubstitution metrics.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast

import train_idrid as hybrid
from eyepacs_hybrid_ordinal.utils import load_config, resolve_device, save_checkpoint, save_json, set_seed

PROTOCOL = "official_train_plus_test_no_validation_final_epoch_resubstitution"
NOTICE = (
    "WARNING: Official TEST images and labels are included in TRAINING. "
    "Final evaluation on those same images is NOT an independent test result."
)


def fit_all(
    training, manifest, cfg, output, device, *, protocol=PROTOCOL, official_test_used_for_training=True,
):
    """Train the supplied pool without validation; record its actual split protocol."""
    run_metadata = {
        "protocol": protocol, "official_test_used_for_training": official_test_used_for_training,
        "evaluation_is_independent": not official_test_used_for_training,
    }
    set_seed(int(cfg["seed"]))
    loader = hybrid.make_loader(training, manifest, cfg, training=True)
    component_options = {}
    if cfg["model"].get("balanced_softmax", False):
        class_counts = [int((training["_label"] == grade).sum()) for grade in range(int(cfg["model"]["num_classes"]))]
        component_options["classification_class_counts"] = class_counts
        balanced_metadata = {
            "enabled": True, "class_counts": class_counts, "counts_source": "actual_training_pool_after_test_copy_exclusion",
            "classification_weight": float(cfg["loss"]["classification_weight"]),
            "prediction_source": "round_and_clip_regression_prediction_not_classification_argmax",
            "source": cfg.get("provenance", {}),
        }
        run_metadata["balanced_softmax"] = balanced_metadata
        save_json(output / "balanced_softmax.json", balanced_metadata)
        print(
            f"Balanced Softmax: weight={cfg['loss']['classification_weight']} | "
            f"TRAIN-only class counts={class_counts} | final predictions=regression", flush=True,
        )
    model, margins, criteria, parameters, optimizer, scaler, amp = hybrid.build_components(cfg, device, **component_options)
    scheduler = hybrid.build_scheduler(optimizer, cfg)
    options = cfg["loss"]
    initial = hybrid.margins_record(margins)
    initial["initialization"] = {
        "method": "uniform" if options.get("margin_init_min") is not None else "fixed",
        "uniform_range": [options.get("margin_init_min"), options.get("margin_init_max")],
        "seed": int(cfg["seed"]), "margin_lr": float(options["margin_lr"]),
    }
    save_json(output / "initial_margins.json", initial)
    print(f"Initial margins: {initial.get('values')} | margin_lr={options['margin_lr']}", flush=True)
    print(
        f"Training pool={len(training)} images | batches/epoch={len(loader)} | "
        "validation=NONE | scheduler=train.rmse | early stopping=OFF | checkpoint=final epoch",
        flush=True,
    )
    phase = "phase1_joint" if margins is not None else "fixed"
    previous, stable_epochs, freeze_reason = None, 0, None
    history = []
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        phase_label = {
            "phase1_joint": "Phase 1: joint", "phase2_frozen": "Phase 2: frozen", "fixed": "Fixed margins",
        }[phase]
        learning_rates = [float(group["lr"]) for group in optimizer.param_groups]
        print(
            f"[{phase_label}] Epoch {epoch}/{cfg['train']['epochs']} | "
            f"network_lr={learning_rates[0]:.2e} | "
            f"margins={'trainable' if phase == 'phase1_joint' else 'fixed'} | NO validation",
            flush=True,
        )
        metrics = hybrid.train_one_epoch(
            model, loader, loader, optimizer, scaler, device, amp,
            criteria, margins, parameters, cfg, epoch, phase=phase_label,
        )
        if not all(math.isfinite(float(metrics[name])) for name in ("total", "pcol", "scol", "mmnp", "rmse")):
            raise FloatingPointError("Non-finite training losses; stopping without evaluation.")
        scheduler.step(float(metrics["rmse"]))
        change = 0.0 if phase == "phase2_frozen" else None
        if phase == "phase1_joint":
            current = margins.margin_values().detach().clone()
            change = float((current - previous).abs().max()) if previous is not None else None
            stable_epochs = stable_epochs + 1 if change is not None and change <= float(options["margin_convergence_tolerance"]) else 0
            previous = current
        margin_state = {
            **hybrid.margins_record(margins), "phase": phase, "freeze_reason": freeze_reason,
            "stable_epochs": stable_epochs, "max_change": change,
        }
        if margins is not None:
            margin_state["initial_values"] = initial["values"]
            margin_state["delta_from_initial"] = [
                value - start for value, start in zip(margin_state["values"], initial["values"])
            ]
        payload = {
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "config": cfg,
            "ordinal_margins": margins.state_dict() if margins is not None else None,
            "margin_state": margin_state, "train_metrics": metrics,
            **run_metadata, "batches_per_epoch": len(loader), "optimizer_steps": epoch * len(loader),
        }
        save_checkpoint(output / "last.pt", payload)
        record = {
            "epoch": epoch, "phase": phase, "learning_rates": learning_rates,
            "train": metrics, "margins": margin_state,
            "batches_per_epoch": len(loader), "optimizer_steps": epoch * len(loader),
        }
        history.append(record)
        save_json(output / "metrics_history.json", {
            **run_metadata, "scheduler_metric": "train.rmse",
            "checkpoint_selection": "final_epoch", "epochs": history,
        })
        save_json(output / "metrics.json", {
            **record, **run_metadata, "checkpoint_selection": "final_epoch",
        })
        print(
            f"epoch={epoch} phase={phase} {hybrid.loss_breakdown(metrics, cfg)} "
            f"train_accuracy={metrics['accuracy']:.4f} train_mae={metrics['mae']:.4f}",
            flush=True,
        )
        if margins is not None:
            print(
                f"margins={margin_state['values']} delta_from_initial={margin_state['delta_from_initial']} "
                f"max_change={change} stable_epochs={stable_epochs}", flush=True,
            )
        if phase == "phase1_joint":
            converged = epoch >= int(options["margin_phase1_min_epochs"]) and stable_epochs >= int(options["margin_convergence_patience"])
            if converged or epoch >= int(options["margin_phase1_max_epochs"]):
                freeze_reason = "margin_converged" if converged else "max_phase1_epochs"
                save_checkpoint(output / "phase1_last.pt", payload)
                margins.freeze()
                phase = "phase2_frozen"
                optimizer.param_groups[0]["lr"] = float(cfg["train"]["lr"])
                scheduler = hybrid.build_scheduler(optimizer, cfg)
                print(
                    f"[Phase 1 -> Phase 2] After epoch {epoch}: {freeze_reason}. "
                    f"Frozen margins={margin_state['values']}; network_lr and scheduler reset. "
                    "No validation or early stopping.", flush=True,
                )
    save_checkpoint(output / "final.pt", payload)
    summary = {
        "final_epoch": epoch, "final_train": metrics, "margins": margin_state,
        "batches_per_epoch": len(loader), "optimizer_steps": epoch * len(loader),
    }
    if "balanced_softmax" in run_metadata:
        summary["balanced_softmax"] = run_metadata["balanced_softmax"]
    return model, amp, summary


@torch.no_grad()
def evaluate_seen_test(model, frame, manifest, cfg, output, device, amp):
    """Re-evaluate the original test subset, whose labels were used for training."""
    model.eval()
    predictions, labels, ids = [], [], []
    for images, targets, image_ids in hybrid.make_loader(frame, manifest, cfg, training=False):
        with autocast(enabled=amp):
            batch = model(images.to(device))["prediction"].float().cpu()
        if not torch.isfinite(batch).all():
            raise FloatingPointError("Non-finite predictions; no resubstitution metrics will be reported.")
        predictions.extend(batch.tolist())
        labels.extend(targets.tolist())
        ids.extend(image_ids)
    metrics = hybrid.test_metrics(predictions, labels)
    classes = np.rint(predictions).clip(0, 4).astype(int)
    payload = {
        "dataset": "IDRiD", "protocol": PROTOCOL,
        "evaluation_split": "official_test_used_for_training", "checkpoint": "final.pt",
        "official_test_used_for_training": True, "evaluation_is_independent": False,
        "metric_interpretation": "Training-data resubstitution; NOT held-out generalization or a comparable CLOC test result.",
        "synthetic_smoke_test": not manifest["official_split_verified"],
        "accuracy_definition": "correct discrete grade predictions / N (micro accuracy)",
        "mae_definition": "mean(abs(round_and_clip(prediction, 0, 4) - true_grade))",
        "seen_test": metrics, "independent_test": None,
    }
    save_json(output / "evaluation_metrics.json", payload)
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "label", "prediction_continuous", "prediction_class", "absolute_error", "included_in_training_pool"])
        writer.writerows(
            (image_id, label, prediction, int(grade), abs(int(grade) - label), True)
            for image_id, label, prediction, grade in zip(ids, labels, predictions, classes)
        )
    print(NOTICE, flush=True)
    print(
        f"IDRiD TRAIN-SEEN test subset (NOT independent): Accuracy={metrics['accuracy_percent']:.2f}% "
        f"MAE={metrics['mae']:.4f} N={metrics['samples']}", flush=True,
    )
    hybrid.report_per_class(metrics, output, split_label="TRAIN-SEEN test")
    return payload


def run(cfg, *, allow_synthetic=False):
    cfg = copy.deepcopy(cfg)
    if int(cfg["data"]["image_size"]) != 224 or int(cfg["model"]["num_classes"]) != 5:
        raise ValueError("This IDRiD script requires 224x224 input and five DR grades.")
    if float(cfg.get("split", {}).get("val_size", 0.0)) != 0.0:
        raise ValueError("Use val_size: 0.0: this script has NO internal validation.")
    if int(cfg["train"]["epochs"]) < 1:
        raise ValueError("epochs must be positive.")
    if cfg["loss"].get("learnable_ordinal_margins", False):
        minimum = int(cfg["loss"]["margin_phase1_min_epochs"])
        maximum = int(cfg["loss"]["margin_phase1_max_epochs"])
        if not 1 <= minimum <= maximum < int(cfg["train"]["epochs"]):
            raise ValueError("Require 1 <= phase1_min <= phase1_max < epochs so Phase 2 can run.")
    cfg["split"] = {"val_size": 0.0}
    cfg["train"].update(
        scheduler_metric="train.rmse", early_stopping_patience=None, checkpoint_selection="final_epoch",
    )
    output = Path(cfg["train"]["output_dir"]).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty; choose a new --output-dir: {output}")
    print(NOTICE, flush=True)
    frames, manifest = hybrid.load_prepared(Path(cfg["data"]["data_dir"]), allow_synthetic=allow_synthetic)
    training = pd.concat([frames["train"], frames["test"]], ignore_index=True)
    device = resolve_device(cfg["device"])
    output.mkdir(parents=True, exist_ok=True)
    git_rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=hybrid.ROOT, capture_output=True, text=True)
    save_json(output / "run_config.json", {
        "config": cfg, "dataset": manifest, "git_commit": git_rev.stdout.strip(),
        "python": platform.python_version(), "torch": str(torch.__version__), "device": str(device),
        "synthetic_smoke_test": allow_synthetic, "protocol": PROTOCOL,
        "official_test_used_for_training": True, "evaluation_is_independent": False,
    })
    save_json(output / "training_split.json", {
        "train": training["sample_id"].tolist(), "validation": [], "independent_test": [],
        "official_test_used_for_training": frames["test"]["sample_id"].tolist(),
        "source_counts": {name: len(frame) for name, frame in frames.items()},
        "training_samples": len(training), "unique_image_hashes": int(training["sha256"].nunique()),
        "class_counts": [int((training["_label"] == grade).sum()) for grade in range(5)],
        "duplicate_summary": manifest.get("duplicate_summary", {}),
        "duplicate_policy": "Preserve all supplied rows and labels; identical files are not new independent images.",
        "method": "concatenate_official_train_and_test_no_holdout", "protocol": PROTOCOL,
    })
    print(
        f"Model: {cfg['model']['backbone']} | pooling={cfg['model']['pooling']} | "
        f"spatial_attention={cfg['model'].get('spatial_attention', False)} | "
        f"PCOL={cfg['loss']['alpha']} SCOLw={cfg['loss']['beta']} "
        f"RMSE={cfg['loss']['rmse_weight']} MMNP={cfg['loss']['mmnp_weight']}", flush=True,
    )
    model, amp, summary = fit_all(training, manifest, cfg, output, device)
    checkpoint = torch.load(output / "final.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    payload = evaluate_seen_test(model, frames["test"], manifest, cfg, output, device, amp)
    payload.update(summary, checkpoint_selection="final_epoch")
    save_json(output / "evaluation_metrics.json", payload)
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    metrics["seen_test"] = payload["seen_test"]
    save_json(output / "metrics.json", metrics)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hybrid_idrid_all_data_no_spatial.yaml")
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
