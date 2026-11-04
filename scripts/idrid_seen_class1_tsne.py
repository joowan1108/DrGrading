"""Re-infer an existing IDRiD train+test model; show all C1 and correct other classes."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train_idrid as hybrid
from evaluate_idrid_test import build_evaluation_model
from eyepacs_hybrid_ordinal.models import clean_state_dict_keys
from eyepacs_hybrid_ordinal.idrid import digest
from eyepacs_hybrid_ordinal.utils import resolve_device, save_json
from idrid_tsne_views import save_view, save_sweep


def analyze(run, data, out, *, device="cpu", allow_synthetic=False, tsne_options=None, sweep=False):

    run, data, out = [Path(p).expanduser().resolve() for p in [run, data, out]]
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Choose a new output folder: {out}")
    reference = json.loads((run / "evaluation_metrics.json").read_text())
    if reference.get("official_test_used_for_training") is not True:
        raise ValueError("Expected the original train+test model evaluation.")
    checkpoint = torch.load(run / "final.pt", map_location="cpu", weights_only=True)
    if checkpoint.get("official_test_used_for_training") is False:
        raise ValueError("Checkpoint records train-only training; choose the actual train+test run.")
    cfg = copy.deepcopy(checkpoint["config"])
    cfg["data"]["num_workers"] = 0
    frames, manifest = hybrid.load_prepared(data, allow_synthetic=allow_synthetic)
    test = frames["test"]
    truth = test["_label"].to_numpy()
    old = reference.get("seen_test") or reference.get("test")
    split = json.loads((run / "training_split.json").read_text())
    if not set(test["sample_id"]) <= set(split["train"]):
        raise ValueError("Training membership does not include all test images.")
    device = resolve_device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu.")
    torch.set_num_threads(2)
    model = build_evaluation_model(cfg, device)
    model.load_state_dict(clean_state_dict_keys(checkpoint["model"]), strict=True)
    model.eval()
    embeddings, images, scores, targets = [], [], [], []
    with torch.inference_mode():
        for batch, labels, ids in hybrid.make_loader(test, manifest, cfg, training=False):
            result = model(batch.to(device))
            embeddings.append(torch.cat([torch.nn.functional.normalize(result[k].float(), dim=1) for k in ["pcol", "scol"]], dim=1).cpu().numpy())
            images.extend(ids)
            scores.extend(result["prediction"].float().cpu().tolist())
            targets.extend(labels.tolist())
    if images != test["_image_id"].tolist() or targets != truth.tolist():
        raise ValueError("Inference did not visit exactly the requested test rows in order.")
    features = np.concatenate(embeddings)
    scores = np.asarray(scores)
    if not np.isfinite(features).all() or not np.isfinite(scores).all():
        raise ValueError("Non-finite model outputs.")
    pred = np.rint(scores).clip(0, 4).astype(int)
    metrics = hybrid.test_metrics(scores.tolist(), truth.tolist())
    saved = pd.DataFrame({"image": images, "sample_id": test["sample_id"].tolist(), "label": truth,
                          "prediction_continuous": scores, "prediction_class": pred, "correct": pred == truth})
    keep = (truth == 1) | (pred == truth)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "evaluation_metrics.json", {"test": metrics, "official_test_used_for_training": True, "evaluation_is_independent": False, "metrics_source": "fresh checkpoint inference on all official test samples", "checkpoint": str(run / "final.pt"), "checkpoint_sha256": digest(run / "final.pt"), "checkpoint_epoch": checkpoint.get("epoch"), "reference_accuracy_percent": 100*old["accuracy"], "accuracy_difference_pp": metrics["accuracy_percent"]-100*old["accuracy"], "device": str(device), "amp": False, "training_performed": False, "synthetic_smoke_test": allow_synthetic})
    hybrid.report_per_class(metrics, out, split_label="training-seen test")
    saved.to_csv(out / "test_predictions.csv", index=False)
    np.savez_compressed(out / "test_embeddings.npz", embeddings=features, labels=truth, predictions=pred, images=np.asarray(images), selected=keep)
    context = {"embedding_device": str(device), "official_test_used_for_training": True}
    metadata = save_view(features, truth, pred, images, out, scores=scores, context=context, **(tsne_options or {}))
    counts = metadata["selected_class_counts"]
    if sweep:
        save_sweep(features, truth, pred, images, out / "sweep", context=context, **(tsne_options or {}))
    print(f"Accuracy={metrics['accuracy_percent']:.2f}% QWK={metrics['quadratic_weighted_kappa']:.4f} MAE={metrics['mae']:.4f}")
    print(f"Original saved accuracy={100*old['accuracy']:.2f}%; difference={metrics['accuracy_percent']-100*old['accuracy']:+.4f} percentage points")
    print(f"t-SNE counts={counts}; total={int(keep.sum())}; included mistakes={metadata['included_incorrect_samples']}")
    print(f"Saved: {out}")
    return metrics, counts


def replot(archive, output, options=None, sweep=False):
    archive, output = Path(archive).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a new output folder: {output}")
    source = json.loads((archive.parent / "evaluation_metrics.json").read_text())
    if source.get("official_test_used_for_training") is not True:
        raise ValueError("Use the original seen-test analysis folder, including evaluation_metrics.json.")
    with np.load(archive, allow_pickle=False) as stored:
        features, truth, pred, images = [stored[k] for k in ["embeddings", "labels", "predictions", "images"]]
    context = {"source_embeddings": str(archive), "source_embeddings_sha256": digest(archive),
               "official_test_used_for_training": True, "inference_performed": False}
    metadata = save_view(features, truth, pred, images, output, context=context, **(options or {}))
    if sweep:
        save_sweep(features, truth, pred, images, output / "sweep", context=context, **(options or {}))
    print(f"Replotted saved features: {output}", flush=True)
    return metadata


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Analyze the saved train+test IDRiD model; include all C1 and correct predictions of other classes in t-SNE.")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "checkpoints/hybrid_idrid_all_data_gem_no_spatial_seed42")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/idrid")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/idrid_8252_class1_all")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--embeddings", type=Path, help="Replot an existing test_embeddings.npz without loading a checkpoint or data.")
    parser.add_argument("--scaling", choices=["none", "standard", "whiten"], default="none")
    parser.add_argument("--perplexity", type=float, default=30)
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--learning-rate", default="auto")
    parser.add_argument("--early-exaggeration", type=float, default=12)
    parser.add_argument("--point-size", type=float, default=12)
    parser.add_argument("--sweep", action="store_true", help="Save all 12 combinations of three scalings and perplexities 5/10/20/30.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    options = dict(scaling=args.scaling, perplexity=args.perplexity, pca_components=args.pca_components,
                   max_iter=args.max_iter, learning_rate="auto" if args.learning_rate == "auto" else float(args.learning_rate),
                   early_exaggeration=args.early_exaggeration, point_size=args.point_size)
    if args.embeddings is not None:
        return replot(args.embeddings, args.output_dir, options, args.sweep)
    return analyze(args.run_dir, args.data_dir, args.output_dir, device=args.device, tsne_options=options, sweep=args.sweep)


if __name__ == "__main__":
    main()
