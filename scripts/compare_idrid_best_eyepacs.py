"""Train/evaluate the selected non-BS EyePACS configuration under two IDRiD protocols."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import train_idrid as hybrid
from export_spatial_attention import resolve_run_directory
from train_idrid_all_data import fit_all
from train_idrid_fixed_iterations import exclude_test_copies
from eyepacs_hybrid_ordinal.idrid_analysis import save_correct_test_tsne
from eyepacs_hybrid_ordinal.utils import resolve_device, save_json

ROOT = hybrid.ROOT
SOURCE = ROOT / "configs/eyepacs_best_gem_no_spatial_source.json"
PROTOCOLS = {"train_plus_test_seen": True, "train_only_heldout": False}


def read_source(source_config=SOURCE, source_run=None):
    if source_run is not None:
        folder = resolve_run_directory(source_run)
        source_name = folder.name
        checkpoint = folder / "best.pt"
        if checkpoint.is_file():
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
            cfg = saved["config"]
            origin = str(checkpoint)
        else:
            metadata = next((folder / name for name in ["run_config.json", "metrics.json"]
                             if (folder / name).is_file()), None)
            if metadata is None:
                raise FileNotFoundError(f"No saved model configuration in {folder}.")
            cfg = json.loads(metadata.read_text())["config"]
            origin = str(metadata)
    else:
        snapshot = json.loads(Path(source_config).read_text())
        source_name = snapshot["source_run"]
        cfg = snapshot["config"]
        origin = str(Path(source_config).resolve())
    cfg = copy.deepcopy(cfg)
    model, loss = cfg["model"], cfg["loss"]
    if (model.get("balanced_softmax", False) or float(loss.get("classification_weight", 0)) != 0
            or model.get("spatial_attention", False) or model.get("pooling") != "gem"):
        raise ValueError("Expected the non-Balanced-Softmax, GeM, no-Spatial-Attention EyePACS model.")
    if model.get("backbone") != "efficientnet_v2_s" or int(cfg["data"]["image_size"]) != 384:
        raise ValueError("Expected the selected EfficientNetV2-S / 384 EyePACS configuration.")
    return cfg, {"source_run": source_name,
                 "configuration_source": origin, "source_weights_loaded_for_idrid_training": False}


def make_config(source, data_dir, output_dir, *, epochs=None, batches_per_epoch=22, device=None,
                num_workers=None, seed=None):
    cfg = copy.deepcopy(source)
    data = source["data"]
    cfg["data"] = {"data_dir": str(Path(data_dir).resolve()), "image_size": int(data["image_size"]),
                   "normalize": data["normalize"], "batch_size": int(data["batch_size"]),
                   "num_workers": int(data.get("num_workers", 4) if num_workers is None else num_workers),
                   "transform_profile": "eyepacs", "train_batches_per_epoch": int(batches_per_epoch)}
    cfg["split"] = {"val_size": 0.0}
    cfg["model"]["balanced_softmax"] = False
    cfg["loss"]["classification_weight"] = 0.0
    # train_hybrid_ordinal uses the MMNP constructor's pair-sum default.
    # The older IDRiD scripts instead defaulted to pair-mean.
    cfg["loss"]["mmnp_pair_reduction"] = "sum"
    cfg["evaluation"] = {"tsne_correct_only": True, "tsne_seed": int(source["seed"] if seed is None else seed)}
    cfg["seed"] = int(source["seed"] if seed is None else seed)
    if device is not None:
        cfg["device"] = device
    cfg["train"].update(output_dir=str(Path(output_dir).resolve()),
                        epochs=int(source["train"]["epochs"] if epochs is None else epochs),
                        early_stopping_patience=None, scheduler_metric="train.rmse", checkpoint_selection="final_epoch")
    if batches_per_epoch < 1 or not 1 <= cfg["loss"]["margin_phase1_min_epochs"] <= cfg["loss"]["margin_phase1_max_epochs"] < cfg["train"]["epochs"]:
        raise ValueError("Require positive batches/epoch and 1 <= Phase 1 min <= Phase 1 max < total epochs.")
    return cfg


@torch.inference_mode()
def evaluate_pool(model, frame, manifest, cfg, output, device, amp, *, scope, training):
    output.mkdir(parents=True, exist_ok=True)
    if scope == "heldout_test":
        for column in ["sample_id", "sha256", "_pixel_sha256"]:
            if set(frame[column]) & set(training[column]):
                raise ValueError(f"Held-out test overlaps training in {column}.")
    model.eval()
    predictions, targets, images, embeddings = [], [], [], []
    is_test = scope != "training_pool"
    for batch, labels, ids in hybrid.make_loader(frame, manifest, cfg, training=False):
        with torch.autocast(device_type=device.type, enabled=amp):
            result = model(batch.to(device))
        scores = result["prediction"].float().cpu()
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Non-finite predictions; evaluation stopped.")
        predictions.extend(scores.tolist()); targets.extend(labels.tolist()); images.extend(ids)
        if is_test:
            embeddings.append(torch.cat([torch.nn.functional.normalize(result[key].float(), dim=1)
                                         for key in ["pcol", "scol"]], dim=1).cpu())
    if images != frame["_image_id"].tolist() or targets != frame["_label"].tolist():
        raise ValueError("Evaluation did not visit exactly the requested rows once.")
    metrics = hybrid.test_metrics(predictions, targets)
    grades = np.rint(predictions).clip(0, 4).astype(int)
    trained_ids = set(training["sample_id"])
    with (output / "predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "sample_id", "label", "prediction_continuous", "prediction_class", "correct", "included_in_training_pool", "evaluation_scope"])
        writer.writerows((image, sample_id, label, score, int(grade), int(grade) == label,
                          sample_id in trained_ids, scope)
                         for image, sample_id, label, score, grade in zip(images, frame["sample_id"], targets, predictions, grades))
    with (output / "confusion_matrix.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["true_class", *[f"pred_C{k}" for k in range(5)]])
        writer.writerows([k, *row] for k, row in enumerate(metrics["confusion_matrix"]))
    hybrid.report_per_class(metrics, output, split_label=scope)
    payload = {"evaluation_scope": scope, "evaluation_is_independent": scope == "heldout_test",
               "samples": len(frame), "metrics": metrics}
    if is_test:
        payload["tsne_correct_only"] = save_correct_test_tsne(
            output, torch.cat(embeddings).numpy(), predictions, targets, images, frame["sample_id"].tolist(),
            seed=int(cfg["evaluation"]["tsne_seed"]), evaluation_scope=scope,
        )
    save_json(output / "evaluation_metrics.json", payload)
    print(f"[{scope}] N={len(frame)} accuracy={metrics['accuracy_percent']:.2f}% "
          f"macro={100*metrics['macro_accuracy']:.2f}% QWK={metrics['quadratic_weighted_kappa']:.4f} MAE={metrics['mae']:.4f}", flush=True)
    return payload


def write_comparison(output, results, expected):
    rows = []
    for name, result in results.items():
        test = result["test"]
        row = {"protocol": name, "official_test_used_for_training": result["official_test_used_for_training"],
               "evaluation_is_independent": result["evaluation_is_independent"],
               "training_samples": result["training_samples"], "test_samples": test["samples"],
               "training_pool_accuracy_percent": result["training_pool"]["metrics"]["accuracy_percent"],
               "test_accuracy_percent": test["accuracy_percent"], "macro_accuracy_percent": 100*test["macro_accuracy"],
               "qwk": test["quadratic_weighted_kappa"], "mae": test["mae"]}
        for k in range(5):
            row[f"C{k}_accuracy_percent"] = test["per_class"][str(k)]["accuracy_percent"]
            row[f"C{k}_support"] = test["per_class"][str(k)]["support"]
        rows.append(row)
    if rows:
        with (output / "comparison.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    save_json(output / "comparison.json", {"evaluation_status": "complete" if len(results) == expected else "running",
              "results": results})


def run_comparison(source, provenance, *, data_dir, output_dir, device=None, epochs=None,
                   batches_per_epoch=22, num_workers=None, seed=None, protocol="both",
                   skip_completed=False, allow_synthetic=False):
    output = Path(output_dir).resolve()
    cfg = make_config(source, data_dir, output, epochs=epochs, batches_per_epoch=batches_per_epoch,
                      device=device, num_workers=num_workers, seed=seed)
    request = {"config": cfg, "source": provenance, "protocol": protocol}
    signature = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    if output.exists() and any(output.iterdir()):
        recorded = output / "request.json"
        if not skip_completed or not recorded.is_file() or json.loads(recorded.read_text())["signature"] != signature:
            raise FileExistsError(f"Choose a new --output-dir, or use --skip-completed for this exact request: {output}")
    frames, manifest = hybrid.load_prepared(Path(data_dir).resolve(), allow_synthetic=allow_synthetic)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "request.json", {**request, "signature": signature})
    save_json(output / "source_eyepacs_config.json", {"source": provenance, "config": source})
    device_obj = resolve_device(cfg["device"])
    names = list(PROTOCOLS) if protocol == "both" else [protocol]
    results = {}
    for name in names:
        seen, folder = PROTOCOLS[name], output / name
        completed = folder / "evaluation_metrics.json"
        if skip_completed and completed.is_file():
            result = json.loads(completed.read_text())
            if result.get("request_signature") != signature or result.get("evaluation_status") != "complete":
                raise ValueError(f"Existing result does not match this request: {folder}")
            old_split = json.loads((folder / "training_split.json").read_text())
            if (old_split["test_pixel_hashes"] != frames["test"]["_pixel_sha256"].tolist()
                    or old_split["test"] != frames["test"]["sample_id"].tolist()
                    or old_split["test_labels"] != frames["test"]["_label"].tolist()):
                raise ValueError("Test membership, labels or pixels changed since the recorded run.")
            training_now = pd.concat([frames["train"], frames["test"]], ignore_index=True) if seen else exclude_test_copies(frames["train"], frames["test"])[0]
            if (old_split["train"] != training_now["sample_id"].tolist()
                    or old_split["training_pixel_hashes"] != training_now["_pixel_sha256"].tolist()
                    or old_split["training_labels"] != training_now["_label"].tolist()):
                raise ValueError("Training membership, labels or pixels changed since the recorded run.")
            results[name] = result
            write_comparison(output, results, len(names)); continue
        if folder.exists() and any(folder.iterdir()):
            raise FileExistsError(f"Incomplete existing run will not be overwritten: {folder}. Use a new --output-dir.")
        if seen:
            training = pd.concat([frames["train"], frames["test"]], ignore_index=True)
            excluded = frames["train"].iloc[:0]
        else:
            training, excluded = exclude_test_copies(frames["train"], frames["test"])
        test = frames["test"].copy()
        test["_overlaps_train"] = test["_pixel_sha256"].isin(set(training["_pixel_sha256"]))
        folder.mkdir(parents=True, exist_ok=True)
        run_cfg = copy.deepcopy(cfg); run_cfg["train"]["output_dir"] = str(folder)
        tag = "eyepacs_best_train_plus_test_final_epoch_seen_test" if seen else "eyepacs_best_train_only_final_epoch_heldout_test"
        split = {"train": training["sample_id"].tolist(), "validation": [], "test": test["sample_id"].tolist(),
                 "official_test_used_for_training": test["sample_id"].tolist() if seen else [],
                 "training_pixel_hashes": training["_pixel_sha256"].tolist(),
                 "test_pixel_hashes": test["_pixel_sha256"].tolist(),
                 "training_labels": training["_label"].tolist(), "test_labels": test["_label"].tolist(),
                 "excluded_training_test_copies": excluded["sample_id"].tolist()}
        save_json(folder / "training_split.json", split)
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        save_json(folder / "run_config.json", {"config": run_cfg, "dataset": manifest, "source": provenance,
                  "git_commit": revision, "python": platform.python_version(), "torch": str(torch.__version__),
                  "protocol": tag, "official_test_used_for_training": seen, "evaluation_is_independent": not seen,
                  "request_signature": signature, "synthetic_smoke_test": allow_synthetic})
        print(f"\n=== {name}: train={len(training)} test={len(test)} | "
              f"input={run_cfg['data']['image_size']} / GeM / no spatial / no Balanced Softmax ===", flush=True)
        model, amp, summary = fit_all(training, manifest, run_cfg, folder, device_obj,
                                      protocol=tag, official_test_used_for_training=seen)
        train_result = evaluate_pool(model, training, manifest, run_cfg, folder / "training_pool", device_obj, amp,
                                     scope="training_pool", training=training)
        test_result = evaluate_pool(model, test, manifest, run_cfg, folder / "test", device_obj, amp,
                                    scope="training_seen_test" if seen else "heldout_test", training=training)
        result = {"protocol": tag, "request_signature": signature, "evaluation_status": "complete",
                  "official_test_used_for_training": seen, "evaluation_is_independent": not seen,
                  "synthetic_smoke_test": allow_synthetic, "checkpoint": "final.pt", "source": provenance,
                  "training_samples": len(training), "training_pool": train_result, "test": test_result["metrics"],
                  "tsne_correct_test": test_result["tsne_correct_only"], **summary}
        save_json(folder / "evaluation_metrics.json", result)
        results[name] = result
        write_comparison(output, results, len(names))
        del model
        if device_obj.type == "cuda":
            torch.cuda.empty_cache()
    print(f"\nComparison saved: {output / 'comparison.csv'}", flush=True)
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--source-config", type=Path, default=SOURCE)
    source.add_argument("--source-run", type=Path, help="Optional actual EyePACS no-spatial run; reads its saved config.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/idrid")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints/idrid_best_eyepacs_comparison_seed42")
    parser.add_argument("--protocol", choices=["both", *PROTOCOLS], default="both")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, help="Default: source EyePACS configuration (75).")
    parser.add_argument("--batches-per-epoch", type=int, default=22)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--skip-completed", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    source, provenance = read_source(args.source_config, args.source_run)
    return run_comparison(source, provenance, **{key: value for key, value in vars(args).items()
                                                if key not in {"source_config", "source_run"}})


if __name__ == "__main__":
    main()
