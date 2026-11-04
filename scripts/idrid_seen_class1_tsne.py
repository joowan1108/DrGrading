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


def analyze(run, data, out, *, device="cpu", allow_synthetic=False):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

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
    chosen, labels, correct = features[keep], truth[keep], (pred == truth)[keep]
    if len(chosen) < 3 or not np.isfinite(chosen).all() or not np.any(np.ptp(chosen, axis=0) > 0):
        raise ValueError("Insufficient or identical features; no artificial t-SNE coordinates created.")
    pca = PCA(n_components=min(50, len(chosen)-1, chosen.shape[1]), svd_solver="full").fit_transform(chosen)
    xy = TSNE(n_components=2, perplexity=min(30.0, len(chosen)-1), init="pca" if pca.shape[1] >= 2 else "random", learning_rate="auto", method="exact", random_state=42).fit_transform(pca)
    if not np.isfinite(xy).all():
        raise ValueError("Non-finite t-SNE coordinates.")
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "evaluation_metrics.json", {"test": metrics, "official_test_used_for_training": True, "evaluation_is_independent": False, "metrics_source": "fresh checkpoint inference on all official test samples", "checkpoint": str(run / "final.pt"), "checkpoint_sha256": digest(run / "final.pt"), "checkpoint_epoch": checkpoint.get("epoch"), "reference_accuracy_percent": 100*old["accuracy"], "accuracy_difference_pp": metrics["accuracy_percent"]-100*old["accuracy"], "device": str(device), "amp": False, "training_performed": False, "synthetic_smoke_test": allow_synthetic})
    hybrid.report_per_class(metrics, out, split_label="training-seen test")
    saved.to_csv(out / "test_predictions.csv", index=False)
    np.savez_compressed(out / "test_embeddings.npz", embeddings=features, labels=truth, predictions=pred, images=np.asarray(images), selected=keep)
    table = saved.loc[keep].copy()
    table["correct"] = correct
    table["tsne_x"], table["tsne_y"] = xy.T
    table.to_csv(out / "tsne_coordinates.csv", index=False)
    counts = {str(k): int((labels == k).sum()) for k in range(5)}
    save_json(out / "tsne_metadata.json", {"selection": "true Class 1: all; other true classes: correct predictions only; filter before PCA/t-SNE", "selected_class_counts": counts, "selected_samples": int(keep.sum()), "included_incorrect_samples": int((~correct).sum()), "total_test_samples": len(truth), "seed": 42, "embedding": "concat(L2(pcol), L2(scol))", "label_colors": "ground truth", "metrics_scope": "all test samples", "embedding_device": str(device), "official_test_used_for_training": True})
    with plt.rc_context({"font.size": 7, "pdf.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(3.5, 3.1))
        handles = []
        for k, color in enumerate(["#7F7F7F", "#0072B2", "#E69F00", "#CC79A7", "#D55E00"]):
            for good, marker in [(True, "o"), (False, "x")]:
                mask = (labels == k) & (correct == good)
                ax.scatter(xy[mask, 0], xy[mask, 1], c=color, marker=marker, s=20, linewidths=0.7, alpha=0.9)
            handles.append(Line2D([], [], color=color, marker="x" if k == 1 and not correct[labels == k].any() else "o", linestyle="None", markersize=4, label=f"C{k} (n={counts[str(k)]})"))
        handles.append(Line2D([], [], color="black", marker="x", linestyle="None", markersize=4, label="Incorrect C1"))
        ax.set_title("IDRiD test (seen in training)\nC1: all; C0/2/3/4: correct", fontsize=8)
        ax.set(xlabel="t-SNE 1", ylabel="t-SNE 2", xticks=[], yticks=[])
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False, fontsize=6)
        fig.tight_layout(pad=0.8)
        for ext in ["pdf", "png"]:
            fig.savefig(out / f"tsne_class1_all_other_correct.{ext}", dpi=600, bbox_inches="tight")
        plt.close(fig)
    print(f"Accuracy={metrics['accuracy_percent']:.2f}% QWK={metrics['quadratic_weighted_kappa']:.4f} MAE={metrics['mae']:.4f}")
    print(f"Original saved accuracy={100*old['accuracy']:.2f}%; difference={metrics['accuracy_percent']-100*old['accuracy']:+.4f} percentage points")
    print(f"t-SNE counts={counts}; total={int(keep.sum())}; included mistakes={int((~correct).sum())}")
    print(f"Saved: {out}")
    return metrics, counts


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Analyze the saved train+test IDRiD model; include all C1 and correct predictions of other classes in t-SNE.")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "checkpoints/hybrid_idrid_all_data_gem_no_spatial_seed42")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/idrid")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/idrid_8252_class1_all")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    return analyze(args.run_dir, args.data_dir, args.output_dir, device=args.device)


if __name__ == "__main__":
    main()
