from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import recolor_tsne
from eyepacs_hybrid_ordinal.plot_colors import DR_CLASS_COLORS


def test_recolor_keeps_all_coordinates_and_labels_without_fitting_tsne(tmp_path, monkeypatch):
    from matplotlib.axes import Axes
    from matplotlib.colors import to_hex

    # Missing C0 must not shift the C1/C2/C3/C4 colors.
    original = pd.DataFrame({"label": [1, 1, 2, 3, 4], "prediction_class": [0, 1, 2, 3, 4],
                             "tsne_x": [-7.123456789, 0, 3.01, -4, 12], "tsne_y": [1, -2.3, 0, 8.67, -1]})
    source = tmp_path / "tsne_coordinates.csv"
    original.to_csv(source, index=False)
    before = source.read_bytes()
    calls = []
    real_scatter = Axes.scatter
    def scatter(axis, x, y, *args, **kwargs):
        if len(x):
            calls.append((np.asarray(x), np.asarray(y), to_hex(kwargs["c"]), kwargs["marker"]))
        return real_scatter(axis, x, y, *args, **kwargs)
    def forbidden(*args, **kwargs):
        pytest.fail("Color changes must not fit t-SNE or PCA.")
    monkeypatch.setattr(Axes, "scatter", scatter)
    monkeypatch.setattr("sklearn.manifold.TSNE.fit_transform", forbidden)
    monkeypatch.setattr("sklearn.decomposition.PCA.fit_transform", forbidden)
    report = recolor_tsne.run(source, tmp_path / "recolored", title="Existing t-SNE")
    assert source.read_bytes() == before
    assert (tmp_path / "recolored/source_coordinates.csv").read_bytes() == before
    assert report["tsne_fitting_performed"] is False and report["coordinate_values_changed"] is False
    assert report["included_incorrect_samples"] == 1
    for x, y, color, marker in calls:
        for xi, yi in zip(x, y):
            row = original[(original.tsne_x == xi) & (original.tsne_y == yi)].iloc[0]
            assert color == DR_CLASS_COLORS[int(row.label)]
            assert marker == ("o" if row.label == row.prediction_class else "x")
    assert (tmp_path / "recolored/tsne_recolored.pdf").read_bytes().startswith(b"%PDF")
    assert json.loads((tmp_path / "recolored/recolor_metadata.json").read_text())["class_colors"] == {
        str(k): value for k, value in enumerate(DR_CLASS_COLORS)}


def test_palette_matches_original_matplotlib_class_colors():
    from matplotlib import colormaps
    from matplotlib.colors import to_hex
    assert DR_CLASS_COLORS == tuple(to_hex(colormaps["tab10"](k)) for k in range(5))


def test_recolor_rejects_missing_coordinates_instead_of_inventing_them(tmp_path):
    source = tmp_path / "unavailable.csv"
    pd.DataFrame({"label": [1], "tsne_x": [np.nan], "tsne_y": [np.nan]}).to_csv(source, index=False)
    with pytest.raises(ValueError, match="existing finite coordinates"):
        recolor_tsne.run(source, tmp_path / "output")
    assert not (tmp_path / "output").exists()
