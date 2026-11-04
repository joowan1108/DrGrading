"""Keep the original EyePACS t-SNE class colors across all experiment figures."""

# Original Matplotlib tab10 colors, assigned by the true grade rather than plot order.
DR_CLASS_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd")

TSNE_FIGURE_SIZE = (9, 7)
TSNE_PNG_DPI = 200


def draw_tsne_classes(axis, coordinates, labels, *, title, point_size=12):
    """Original EyePACS display: one circular marker and one color per true class."""
    import numpy as np

    labels = np.asarray(labels)
    for grade in sorted(np.unique(labels)):
        keep = labels == grade
        axis.scatter(coordinates[keep, 0], coordinates[keep, 1], s=point_size, alpha=0.7,
                     marker="o", color=DR_CLASS_COLORS[int(grade)], label=f"Class {grade}")
    axis.set(title=title, xlabel="t-SNE 1", ylabel="t-SNE 2")
    axis.set_aspect("auto")
    axis.legend(title="True label", markerscale=1.5)
