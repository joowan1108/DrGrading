"""Label-independent scaling and reproducible t-SNE views of saved IDRiD features."""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd

from eyepacs_hybrid_ordinal.utils import save_json


def save_view(features, targets, predictions, images, output, *, scaling="none", perplexity=30,
              pca_components=50, max_iter=1000, learning_rate="auto", early_exaggeration=12,
              seed=42, point_size=12, scores=None, context=None):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    features, targets, predictions = np.asarray(features), np.asarray(targets), np.asarray(predictions)
    if (features.ndim != 2 or len(features) != len(targets) or len(images) != len(targets)
            or predictions.shape != targets.shape or not np.isfinite(features).all()
            or not np.isin(targets, range(5)).all() or not np.isin(predictions, range(5)).all()):
        raise ValueError("Features, true grades, predictions and images must be finite and aligned.")
    if (scaling not in {"none", "standard", "whiten"} or perplexity <= 0 or pca_components < 1
            or max_iter < 250 or early_exaggeration <= 0 or point_size <= 0
            or (learning_rate != "auto" and (not np.isfinite(learning_rate) or learning_rate <= 0))):
        raise ValueError("Invalid scaling or t-SNE parameters.")
    keep = (targets == 1) | (predictions == targets)
    selected, labels, correct = features[keep].copy(), targets[keep], (predictions == targets)[keep]
    if len(selected) < 3 or not np.any(np.ptp(selected, axis=0) > 0):
        raise ValueError("Insufficient or identical selected features; no coordinates fabricated.")
    original = selected.copy()
    if scaling == "standard":
        selected = StandardScaler().fit_transform(selected)
    reducer = PCA(n_components=min(pca_components, len(selected)-1, selected.shape[1]), svd_solver="full")
    reduced = reducer.fit_transform(selected)
    if scaling == "whiten":
        # Drop numerically null components instead of amplifying them without bound.
        valid = reducer.explained_variance_ > max(float(reducer.explained_variance_.max())*1e-8, 1e-15)
        reduced = reduced[:, valid] / np.sqrt(reducer.explained_variance_[valid])
    if reduced.shape[1] == 0 or not np.isfinite(reduced).all():
        raise ValueError("No finite nonzero PCA components remain.")
    effective_perplexity = min(float(perplexity), float(len(selected)-1))
    options = dict(n_components=2, perplexity=effective_perplexity, init="pca" if reduced.shape[1] >= 2 else "random",
                   learning_rate=learning_rate, early_exaggeration=early_exaggeration, method="exact", random_state=int(seed))
    options["max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"] = int(max_iter)
    estimator = TSNE(**options)
    xy = estimator.fit_transform(reduced)
    if not np.isfinite(xy).all():
        raise ValueError("Non-finite t-SNE coordinates.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame({"image": np.asarray(images)[keep], "label": labels,
                          "prediction_class": predictions[keep], "correct": correct,
                          "tsne_x": xy[:, 0], "tsne_y": xy[:, 1]})
    if scores is not None:
        table["prediction_continuous"] = np.asarray(scores)[keep]
    table.to_csv(output / "tsne_coordinates.csv", index=False)
    np.savez_compressed(output / "tsne_input.npz", selected_embeddings=original, reduced_embeddings=reduced,
                        selected=keep, labels=labels, predictions=predictions[keep])
    counts = {str(k): int((labels == k).sum()) for k in range(5)}
    metadata = {**(context or {}), "selection": "true Class 1: all; other true classes: correct predictions only; filter before scaling/PCA/t-SNE",
                "selected_class_counts": counts, "selected_samples": int(keep.sum()),
                "included_incorrect_samples": int((~correct).sum()), "total_test_samples": len(targets),
                "seed": int(seed), "embedding": "concat(L2(pcol), L2(scol))", "label_colors": "ground truth",
                "metrics_scope": "all test samples", "scaling": scaling, "perplexity_requested": float(perplexity),
                "perplexity": effective_perplexity, "pca_components_requested": int(pca_components),
                "pca_components": reduced.shape[1], "pca_explained_variance_ratio_sum": float(reducer.explained_variance_ratio_.sum()),
                "max_iter": int(max_iter), "learning_rate": learning_rate, "early_exaggeration": early_exaggeration,
                "point_size": point_size, "kl_divergence": float(estimator.kl_divergence_),
                "whitening_null_component_relative_threshold": 1e-8 if scaling == "whiten" else None}
    save_json(output / "tsne_metadata.json", metadata)
    with plt.rc_context({"font.size": 7, "pdf.fonttype": 42}):
        figure, axis = plt.subplots(figsize=(3.5, 3.1))
        handles = []
        for grade, color in enumerate(["#7F7F7F", "#0072B2", "#E69F00", "#CC79A7", "#D55E00"]):
            for good, marker in [(True, "o"), (False, "x")]:
                mask = (labels == grade) & (correct == good)
                axis.scatter(xy[mask, 0], xy[mask, 1], c=color, marker=marker, s=point_size, linewidths=0.5, alpha=0.85)
            handles.append(Line2D([], [], color=color, marker="x" if grade == 1 and not correct[labels == grade].any() else "o",
                                  linestyle="None", markersize=4, label=f"C{grade} (n={counts[str(grade)]})"))
        handles.append(Line2D([], [], color="black", marker="x", linestyle="None", markersize=4, label="Incorrect C1"))
        axis.set_title("IDRiD test (seen in training)\nC1: all; C0/2/3/4: correct", fontsize=8)
        axis.set(xlabel="t-SNE 1", ylabel="t-SNE 2", xticks=[], yticks=[])
        axis.set_aspect("equal", adjustable="datalim")
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False, fontsize=6)
        figure.tight_layout(pad=0.8)
        for extension in ["pdf", "png"]:
            figure.savefig(output / f"tsne_class1_all_other_correct.{extension}", dpi=600, bbox_inches="tight")
        plt.close(figure)
    return metadata


def save_sweep(features, targets, predictions, images, output, **options):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    output = Path(output)
    views = []
    for scaling in ["none", "standard", "whiten"]:
        for perplexity in [5, 10, 20, 30]:
            folder = output / f"{scaling}_p{perplexity}"
            metadata = save_view(features, targets, predictions, images, folder,
                                 **{**options, "scaling": scaling, "perplexity": perplexity})
            views.append({"folder": str(folder), "scaling": scaling, "perplexity": perplexity, "metadata": metadata})
            print(f"[t-SNE sweep] {scaling} / perplexity {perplexity}", flush=True)
    figure, axes = plt.subplots(3, 4, figsize=(12, 9.4), constrained_layout=True)
    for axis, view in zip(axes.flat, views):
        axis.imshow(plt.imread(Path(view["folder"]) / "tsne_class1_all_other_correct.png"))
        axis.set_title(f"{view['scaling']} | perplexity={view['perplexity']}", fontsize=9)
        axis.axis("off")
    figure.savefig(output / "tsne_sweep.png", dpi=200, bbox_inches="tight")
    figure.savefig(output / "tsne_sweep.pdf", bbox_inches="tight")
    plt.close(figure)
    save_json(output / "tsne_sweep.json", {"views": views, "selection_based_on_labels_for_coordinates": False})
    return views
