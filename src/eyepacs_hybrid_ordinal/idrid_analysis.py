"""Compact t-SNE of correctly classified official-test samples only."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .utils import save_json
from .plot_colors import DR_CLASS_COLORS


def save_correct_test_tsne(
    output_dir: Path, embeddings, predictions, targets, image_ids, sample_ids, *, seed=42, evaluation_scope=None,
) -> dict:
    features = np.asarray(embeddings, dtype=np.float32)
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    n = len(targets)
    if features.ndim != 2 or features.shape[0] != n or features.shape[1] == 0:
        raise ValueError("Test embeddings must be a nonempty-feature matrix aligned with labels.")
    if any(len(values) != n for values in (predictions, image_ids, sample_ids)):
        raise ValueError("Test features, predictions and IDs must refer to the same samples.")
    if not np.isfinite(features).all() or not np.isfinite(predictions).all():
        raise ValueError("Non-finite test features/predictions cannot be plotted.")
    if not np.isin(targets, range(5)).all() or not all(str(s).startswith("test:") for s in sample_ids):
        raise ValueError("The t-SNE input must contain official test samples with grades 0–4.")
    classes = np.rint(predictions).clip(0, 4).astype(np.int64)
    correct = classes == targets
    # Filter BEFORE PCA and t-SNE fitting; mistakes cannot influence coordinates.
    selected = features[correct]
    labels = targets[correct]
    selected_images = np.asarray(image_ids, dtype=str)[correct]
    selected_ids = np.asarray(sample_ids, dtype=str)[correct]
    selected_predictions = predictions[correct]
    selected_classes = classes[correct]
    count = len(labels)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "tsne_test_correct_only"
    metadata = {
        "status": "pending", "selection": "prediction_class == true_grade, official_test only",
        "metric_scope": "Accuracy/Macro/QWK/MAE/per-class metrics use ALL test samples, not this subset",
        "embedding": "concat(l2_normalized_pcol, l2_normalized_scol)",
        "embedding_dimension": int(features.shape[1]), "seed": int(seed),
        "total_test_samples": n, "correct_samples": count, "excluded_incorrect_samples": n - count,
        "test_class_counts": {str(k): int((targets == k).sum()) for k in range(5)},
        "correct_class_counts": {str(k): int((labels == k).sum()) for k in range(5)},
        "pca_components": None, "perplexity": None,
        "class_colors": {str(k): color for k, color in enumerate(DR_CLASS_COLORS)},
        "image_file": f"{stem}.png", "pdf_file": f"{stem}.pdf",
        "coordinates_file": f"{stem}.csv", "embeddings_file": f"{stem}_embeddings.npz",
        "metadata_file": f"{stem}_metadata.json",
        "interpretation": "Correct-only qualitative visualization; does not represent separation of all test samples.",
    }
    if evaluation_scope is not None:
        if evaluation_scope not in {"heldout_test", "training_seen_test"}:
            raise ValueError("Unknown t-SNE evaluation scope.")
        metadata.update(evaluation_scope=evaluation_scope,
                        official_test_used_for_training=evaluation_scope == "training_seen_test",
                        evaluation_is_independent=evaluation_scope == "heldout_test")
    np.savez_compressed(
        output_dir / metadata["embeddings_file"], embeddings=selected, labels=labels,
        sample_ids=selected_ids, images=selected_images,
        predictions_continuous=selected_predictions, predictions_class=selected_classes,
    )
    coordinates = None
    if count < 3:
        metadata.update(status="skipped_insufficient_correct_samples", reason="t-SNE requires at least 3 correctly classified samples; no coordinates were fabricated.")
    elif not np.any(np.ptp(selected, axis=0) > 0):
        metadata.update(status="skipped_identical_embeddings", reason="All selected feature vectors are identical; no meaningful t-SNE can be fitted.")
    else:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE

        components = min(50, selected.shape[1], count - 1)
        reduced = PCA(n_components=components, svd_solver="full").fit_transform(selected)
        perplexity = min(30.0, float(count - 1))
        coordinates = TSNE(
            n_components=2, perplexity=perplexity, learning_rate="auto",
            init="pca" if components >= 2 else "random", method="exact", random_state=int(seed),
        ).fit_transform(reduced)
        if not np.isfinite(coordinates).all():
            raise ValueError("t-SNE returned non-finite coordinates.")
        metadata.update(status="created", pca_components=components, perplexity=perplexity)
    with (output_dir / metadata["coordinates_file"]).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "image", "label", "prediction_class", "prediction_continuous", "tsne_x", "tsne_y"])
        for i in range(count):
            point = coordinates[i].tolist() if coordinates is not None else ["", ""]
            writer.writerow([selected_ids[i], selected_images[i], int(labels[i]), int(selected_classes[i]), float(selected_predictions[i]), *point])

    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    with plt.rc_context({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        figure, axis = plt.subplots(figsize=(3.5, 3.0))
        scope_label = {"heldout_test": "held-out test", "training_seen_test": "training-seen test"}.get(evaluation_scope, "test")
        axis.set_title(f"IDRiD {scope_label}: correct predictions\nn = {count}/{n}", pad=7)
        if coordinates is not None:
            handles = []
            for grade, color in enumerate(DR_CLASS_COLORS):
                mask = labels == grade
                axis.scatter(coordinates[mask, 0], coordinates[mask, 1], c=color, s=14,
                             alpha=0.85, linewidths=0.25, edgecolors="white")
                handles.append(Line2D([], [], color=color, marker="o", linestyle="None", markersize=4,
                                      label=f"C{grade} (n={int(mask.sum())})"))
            axis.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13),
                        ncol=3, frameon=False, handletextpad=0.3, columnspacing=0.8, fontsize=6.5)
            axis.set(xlabel="t-SNE 1", ylabel="t-SNE 2", xticks=[], yticks=[])
            axis.spines[["top", "right"]].set_visible(False)
        else:
            axis.text(0.5, 0.5, "t-SNE unavailable\n" + (
                "Fewer than 3 correct test samples" if count < 3 else "Selected features are identical"
            ), ha="center", va="center", transform=axis.transAxes)
            axis.set_axis_off()
        figure.tight_layout(pad=0.8)
        figure.savefig(output_dir / metadata["image_file"], dpi=600, bbox_inches="tight", facecolor="white")
        figure.savefig(output_dir / metadata["pdf_file"], bbox_inches="tight", facecolor="white")
        plt.close(figure)
    save_json(output_dir / metadata["metadata_file"], metadata)
    print(f"[TEST t-SNE] {metadata['status']}: correct={count}/{n}, per class={metadata['correct_class_counts']}", flush=True)
    return metadata
