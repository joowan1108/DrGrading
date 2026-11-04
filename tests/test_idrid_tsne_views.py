from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import idrid_seen_class1_tsne as analysis
from idrid_tsne_views import save_view, save_sweep
from eyepacs_hybrid_ordinal.plot_colors import DR_CLASS_COLORS


def arrays():
    labels = np.repeat(np.arange(5), 4)
    predictions = labels.copy(); predictions[0] = 2; predictions[4] = 0
    features = np.random.default_rng(8).normal(size=(20, 8)).astype(np.float32)
    features[:, 0] *= 100
    features[:, -1] = 7  # A null PCA direction must not explode under whitening.
    images = np.array([f"test:{i}" for i in range(20)])
    return features, labels, predictions, images


@pytest.mark.parametrize("scaling", ["none", "standard", "whiten"])
def test_scaling_uses_only_selected_features_and_keeps_predictions_and_labels(tmp_path, scaling, monkeypatch):
    from sklearn.manifold import TSNE
    methods = []
    class RecordedTSNE(TSNE):
        def fit_transform(self, features, *args, **kwargs):
            methods.append(self.method)
            assert self.method == "barnes_hut"
            return super().fit_transform(features, *args, **kwargs)
    monkeypatch.setattr("sklearn.manifold.TSNE", RecordedTSNE)
    features, labels, predictions, images = arrays()
    untouched = features.copy()
    report = save_view(features, labels, predictions, images, tmp_path, scaling=scaling, perplexity=10,
                       pca_components=5, max_iter=300)
    keep = (labels == 1) | (predictions == labels)
    assert report["selected_samples"] == 19 and report["included_incorrect_samples"] == 1
    assert report["selected_class_counts"]["1"] == 4
    assert report["scaling"] == scaling and report["perplexity"] == 10
    assert report["method"] == "barnes_hut" and methods == ["barnes_hut"]
    assert report["pca_svd_solver"] == "auto" and report["marker"] == "o"
    assert report["point_alpha"] == 0.7 and report["figure_size_inches"] == [9, 7]
    assert report["class_colors"] == {str(k): color for k, color in enumerate(DR_CLASS_COLORS)}
    with np.load(tmp_path / "tsne_input.npz") as saved:
        np.testing.assert_array_equal(saved["selected_embeddings"], features[keep])
        np.testing.assert_array_equal(saved["predictions"], predictions[keep])
        assert np.isfinite(saved["reduced_embeddings"]).all()
        if scaling == "whiten":
            np.testing.assert_allclose(saved["reduced_embeddings"].var(axis=0, ddof=1), 1, atol=1e-5)
    np.testing.assert_array_equal(features, untouched)
    assert (tmp_path / "tsne_class1_all_other_correct.pdf").read_bytes().startswith(b"%PDF")


def test_cached_replot_never_loads_checkpoint_or_test_images(tmp_path, monkeypatch):
    original = tmp_path / "original"; original.mkdir()
    features, labels, predictions, images = arrays()
    archive = original / "test_embeddings.npz"
    np.savez_compressed(archive, embeddings=features, labels=labels, predictions=predictions, images=images)
    metrics = original / "evaluation_metrics.json"
    metrics.write_text(json.dumps({"official_test_used_for_training": True, "test": {"accuracy": 0.8252}}))
    before = archive.read_bytes(), metrics.read_bytes()
    def forbidden(*args, **kwargs):
        pytest.fail("Replot must not run inference or load test images.")
    monkeypatch.setattr(analysis, "analyze", forbidden)
    monkeypatch.setattr(analysis.hybrid, "load_prepared", forbidden)
    monkeypatch.setattr(analysis, "build_evaluation_model", forbidden)
    result = analysis.main(["--embeddings", str(archive), "--output-dir", str(tmp_path / "replot"),
                            "--scaling", "standard", "--perplexity", "5", "--max-iter", "300"])
    assert result["inference_performed"] is False and result["scaling"] == "standard"
    assert (archive.read_bytes(), metrics.read_bytes()) == before
    assert not (tmp_path / "replot/evaluation_metrics.json").exists()  # Plotting cannot change QWK/accuracy.


def test_sweep_exports_all_scaling_perplexity_combinations(tmp_path, monkeypatch):
    import idrid_tsne_views as views
    from PIL import Image
    calls = []
    def render(features, labels, predictions, images, folder, **options):
        folder.mkdir(parents=True)
        Image.new("RGB", (30, 25), "white").save(folder / "tsne_class1_all_other_correct.png")
        calls.append((options["scaling"], options["perplexity"]))
        return {"selected_samples": 19}
    monkeypatch.setattr(views, "save_view", render)
    save_sweep(*arrays(), tmp_path, max_iter=300)
    assert calls == [(s, p) for s in ["none", "standard", "whiten"] for p in [5, 10, 20, 30]]
    assert len(json.loads((tmp_path / "tsne_sweep.json").read_text())["views"]) == 12
    assert (tmp_path / "tsne_sweep.pdf").read_bytes().startswith(b"%PDF")


@pytest.mark.parametrize("options", [{"perplexity": 0}, {"max_iter": 100}, {"scaling": "invalid"}])
def test_rejects_invalid_tsne_options(tmp_path, options):
    with pytest.raises(ValueError, match="Invalid scaling"):
        save_view(*arrays(), tmp_path, **options)


def test_default_idrid_matches_actual_eyepacs_coordinates_and_uses_circles_for_c1_errors(tmp_path, monkeypatch):
    import pandas as pd
    from PIL import Image
    from matplotlib.axes import Axes
    from train_hybrid_ordinal import save_outer_test_tsne

    features, truth, pred, images = arrays()
    keep = (truth == 1) | (truth == pred)
    original = tmp_path / "eyepacs_reference"; original.mkdir()
    candidate = tmp_path / "idrid"
    reference = save_outer_test_tsne(original, features[keep], pred[keep].astype(float).tolist(),
                                     truth[keep].tolist(), images[keep].tolist(), seed=42)
    calls = []
    real_scatter = Axes.scatter
    def scatter(axis, x, y, *args, **kwargs):
        calls.append(kwargs)
        return real_scatter(axis, x, y, *args, **kwargs)
    monkeypatch.setattr(Axes, "scatter", scatter)
    metadata = save_view(features, truth, pred, images, candidate)
    a = pd.read_csv(original / reference["coordinates_file"])
    b = pd.read_csv(candidate / "tsne_coordinates.csv")
    np.testing.assert_allclose(a[["tsne_x", "tsne_y"]], b[["tsne_x", "tsne_y"]], rtol=1e-6, atol=1e-6)
    assert metadata["max_iter"] == 1000 and metadata["scaling"] == "none"
    assert metadata["perplexity"] == min(30, len(b)-1)
    assert metadata["included_incorrect_samples"] == 1
    assert (b.label == 1).sum() == 4  # Removing the X must not remove the C1 mistakes.
    assert len(calls) == 5 and all(call["marker"] == "o" and call["alpha"] == 0.7 and call["s"] == 12 for call in calls)
    assert Image.open(original / reference["image_file"]).size == Image.open(candidate / "tsne_class1_all_other_correct.png").size == (1800, 1400)


def test_bs_and_non_bs_cli_share_original_eyepacs_tsne_defaults():
    import train_idrid_seen_balanced_softmax as runner
    for args in [analysis.parse_args([]), runner.parse_args([])]:
        assert args.scaling == "none" and args.perplexity == 30 and args.max_iter == 1000
