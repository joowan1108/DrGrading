"""Render saved t-SNE coordinates with the original EyePACS class colors."""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eyepacs_hybrid_ordinal.plot_colors import DR_CLASS_COLORS
from eyepacs_hybrid_ordinal.utils import save_json


def run(coordinates, output_dir, *, title="t-SNE", point_size=12):
    coordinates, output = Path(coordinates).expanduser().resolve(), Path(output_dir).expanduser().resolve()
    table = pd.read_csv(coordinates)
    if not {"label", "tsne_x", "tsne_y"} <= set(table.columns):
        raise ValueError("Coordinate CSV must include label, tsne_x and tsne_y.")
    labels = pd.to_numeric(table["label"], errors="raise").to_numpy()
    xy = table[["tsne_x", "tsne_y"]].to_numpy(dtype=float)
    if len(table) < 1 or not np.isin(labels, range(5)).all() or not np.isfinite(xy).all():
        raise ValueError("Need existing finite coordinates with true grades 0-4; no t-SNE is recalculated.")
    if point_size <= 0 or not np.isfinite(point_size):
        raise ValueError("point-size must be positive and finite.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a new output folder: {output}")
    correct = np.ones(len(table), dtype=bool)
    if "prediction_class" in table:
        correct = table["prediction_class"].to_numpy() == labels
    elif "prediction" in table:
        correct = np.rint(table["prediction"].to_numpy(dtype=float)).clip(0, 4) == labels
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.size": 7, "pdf.fonttype": 42}):
        figure, axis = plt.subplots(figsize=(3.5, 3.1))
        handles = []
        for grade, color in enumerate(DR_CLASS_COLORS):
            for matched, marker in [(True, "o"), (False, "x")]:
                keep = (labels == grade) & (correct == matched)
                axis.scatter(xy[keep, 0], xy[keep, 1], c=color, marker=marker, s=point_size, alpha=0.85, linewidths=0.5)
            handles.append(Line2D([], [], color=color, marker="o", linestyle="None", markersize=4,
                                  label=f"C{grade} (n={int((labels == grade).sum())})"))
        if not correct.all():
            handles.append(Line2D([], [], color="black", marker="x", linestyle="None", markersize=4, label="Incorrect"))
        axis.set(title=title, xlabel="t-SNE 1", ylabel="t-SNE 2", xticks=[], yticks=[])
        axis.set_aspect("equal", adjustable="datalim")
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False, fontsize=6)
        figure.tight_layout(pad=0.8)
        for extension in ["png", "pdf"]:
            figure.savefig(output / f"tsne_recolored.{extension}", dpi=600, bbox_inches="tight")
        plt.close(figure)
    shutil.copyfile(coordinates, output / "source_coordinates.csv")
    metadata = {"source_coordinates": str(coordinates), "source_coordinates_sha256": hashlib.sha256(coordinates.read_bytes()).hexdigest(),
                "coordinate_values_changed": False, "samples": len(table), "included_incorrect_samples": int((~correct).sum()),
                "class_colors": {str(k): color for k, color in enumerate(DR_CLASS_COLORS)}, "training_performed": False,
                "inference_performed": False, "tsne_fitting_performed": False, "point_size": point_size, "title": title}
    save_json(output / "recolor_metadata.json", metadata)
    print(f"Saved original-color t-SNE without recalculating coordinates: {output}")
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="t-SNE")
    parser.add_argument("--point-size", type=float, default=12)
    args = parser.parse_args(argv)
    return run(**vars(args))


if __name__ == "__main__":
    main()
