"""Export the actual CBAM spatial gate for one labelled EyePACS image."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eyepacs_hybrid_ordinal.data import load_eyepacs_dataframe, resolve_image_path
from eyepacs_hybrid_ordinal.metrics import ordinal_class_predictions
from eyepacs_hybrid_ordinal.models import HybridOrdinalNet, clean_state_dict_keys
from eyepacs_hybrid_ordinal.splitting import build_nested_split_indices, build_split_indices
from eyepacs_hybrid_ordinal.transforms import CropBlackBorder, make_eval_transform
from eyepacs_hybrid_ordinal.utils import load_config, resolve_device, save_json


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def resolve_run_directory(value: str | Path) -> Path:
    """Locate the exact named run in this checkout or sibling project checkouts."""
    requested = Path(value).expanduser()
    if requested.is_dir():
        return requested.resolve()
    matches = set()
    if not requested.is_absolute():
        name = requested.name
        candidates = [ROOT / requested, ROOT / "checkpoints" / name, ROOT.parent / name]
        candidates += [project / "checkpoints" / name for project in ROOT.parent.iterdir() if project.is_dir()]
        matches = {candidate.resolve() for candidate in candidates if candidate.is_dir()}
    if len(matches) == 1:
        found = matches.pop()
        print(f"Resolved exact run name: {found}", flush=True)
        return found
    if len(matches) > 1:
        raise ValueError("Several folders have this run name; pass an absolute --run-dir:\n" + "\n".join(map(str, sorted(matches))))
    raise FileNotFoundError(
        f"Run directory not found: {requested.resolve()}. Use the actual EyePACS Spatial Attention "
        "folder, e.g. hybrid_eyepacs_efficientnet_v2_s_384_gem_spatial_cloc110-2, "
        "or pass --checkpoint /actual/path/best.pt."
    )


def checkpoint_and_config(args):
    path = (Path(args.checkpoint).expanduser() if args.checkpoint
            else resolve_run_directory(args.run_dir) / "best.pt").resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Use the Spatial Attention run's best.pt, "
            "or pass its exact path with --checkpoint."
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = clean_state_dict_keys(checkpoint.get("model", checkpoint.get("state_dict", checkpoint)))
    if "spatial_attention.conv.weight" not in state:
        raise ValueError(
            "This checkpoint has no trained Spatial Attention weights. "
            "Choose a spatial_attention=true run; a no_spatial/GeM-only checkpoint cannot export this map."
        )
    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = checkpoint.get("config")
        if not isinstance(cfg, dict):
            recorded = path.parent / "run_config.json"
            cfg = json.loads(recorded.read_text())["config"] if recorded.is_file() else None
    if not isinstance(cfg, dict):
        raise ValueError("No saved configuration found. Pass the training configuration with --config.")
    cfg = copy.deepcopy(cfg)
    if not cfg["model"].get("spatial_attention", False):
        raise ValueError("The selected configuration disables Spatial Attention. Use the matching training config.")
    checkpoint_root = next((parent.parent for parent in path.parents if parent.name == "checkpoints"), ROOT)
    for key in ["labels_csv", "image_root"]:
        if getattr(args, key):
            cfg["data"][key] = str(Path(getattr(args, key)).expanduser().resolve())
        elif key in cfg["data"]:
            configured = Path(cfg["data"][key]).expanduser()
            cfg["data"][key] = str(configured if configured.is_absolute() else checkpoint_root / configured)
    return path, checkpoint, state, cfg


def select_sample(cfg, *, split="val", class_id=3, sample_index=0, image_id=None):
    data, splits = cfg["data"], cfg.get("split", {})
    frame = load_eyepacs_dataframe(
        project_path(data["labels_csv"]), image_col=data.get("image_col"),
        label_col=data.get("label_col"), subject_col=data.get("subject_col"),
        max_samples=data.get("max_samples"),
    )
    options = dict(seed=int(cfg.get("seed", 42)), val_size=float(splits.get("val_size", 0.2)),
                   subject_independent=bool(splits.get("subject_independent", True)))
    folds = int(splits.get("num_folds", 0) or 0)
    if folds > 1:
        _, val_idx, test_idx = build_nested_split_indices(
            frame, **options, num_folds=folds, fold_index=int(splits.get("fold_index", 0)))
    else:
        _, val_idx = build_split_indices(frame, **options, num_folds=0)
        test_idx = None
    if split == "test" and test_idx is None:
        raise ValueError("This run has no configured outer test split. Use --split val.")
    selected_split = frame.iloc[val_idx if split == "val" else test_idx]
    candidates = selected_split.loc[selected_split["_label"] == int(class_id)].sort_values("_image_id")
    candidate_count = len(candidates)
    if image_id is not None:
        candidates = candidates.loc[candidates["_image_id"] == str(image_id)]
        if candidates.empty:
            raise ValueError(f"Image {image_id!r} is not a ground-truth Class {class_id} image in {split}.")
        row = candidates.iloc[0]
    else:
        if not 0 <= sample_index < candidate_count:
            raise ValueError(f"Class {class_id} has {candidate_count} images in {split}; --sample-index is out of range.")
        row = candidates.iloc[sample_index]
    image_path = resolve_image_path(project_path(data["image_root"]), str(row["_image_id"])).resolve()
    return row, image_path, candidate_count


def crop_box(image):
    """Use exactly the threshold and padding of the inference border crop."""
    crop = CropBlackBorder()
    yy, xx = np.where(np.asarray(image.convert("L")) > crop.threshold)
    if not len(xx):
        return 0, 0, image.width, image.height
    return (max(int(xx.min())-crop.padding, 0), max(int(yy.min())-crop.padding, 0),
            min(int(xx.max())+crop.padding+1, image.width), min(int(yy.max())+crop.padding+1, image.height))


def save_visuals(output_dir, original, model_input, gate, *, image_id, label, prediction, score, alpha):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    native = gate.numpy()
    size = int(model_input.shape[-1])
    enlarged = F.interpolate(gate[None, None], (size, size), mode="bilinear", align_corners=False)[0, 0].numpy()
    view = Image.fromarray((model_input.permute(1, 2, 0).numpy()*255).round().clip(0, 255).astype(np.uint8))
    vmin, vmax = float(native.min()), float(native.max())
    if vmax-vmin < 1e-8:
        vmin, vmax = 0.0, 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("turbo")
    heatmap = Image.fromarray(cmap(norm(enlarged), bytes=True)[..., :3])
    overlay = Image.blend(view, heatmap, alpha)
    box = crop_box(original)
    crop = original.crop(box)
    full_gate = F.interpolate(gate[None, None], (crop.height, crop.width), mode="bilinear", align_corners=False)[0, 0].numpy()
    crop_overlay = Image.blend(crop, Image.fromarray(cmap(norm(full_gate), bytes=True)[..., :3]), alpha)
    original_overlay = original.copy()
    original_overlay.paste(crop_overlay, box[:2])

    np.save(output_dir / "spatial_attention.npy", native, allow_pickle=False)
    np.save(output_dir / "spatial_attention_input.npy", enlarged, allow_pickle=False)
    view.save(output_dir / "input.png")
    heatmap.save(output_dir / "spatial_attention.png")
    overlay.save(output_dir / "overlay_input.png")
    original_overlay.save(output_dir / "overlay_original.png")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), constrained_layout=True)
    axes[0].imshow(view)
    axes[0].set_title("EyePACS input")
    plotted = axes[1].imshow(native, cmap=cmap, norm=norm, interpolation="nearest")
    axes[1].set_title(f"Spatial gate ({native.shape[0]} x {native.shape[1]})")
    axes[2].imshow(overlay)
    axes[2].set_title("Attention overlay")
    for ax in axes:
        ax.axis("off")
    fig.colorbar(plotted, ax=axes[1:], fraction=0.035, pad=0.02, label="Spatial weight")
    fig.suptitle(f"{image_id} | ground truth {label} | prediction {prediction} (score {score:.3f})", fontsize=11)
    fig.savefig(output_dir / "comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    return {"crop_box_xyxy": list(box), "display_limits": [vmin, vmax],
            "display_scale": "linear in raw sigmoid weights, limits shown on the colorbar",
            "interpolation": "bilinear, align_corners=False", "overlay_alpha": alpha}


def run(args):
    path, checkpoint, state, cfg = checkpoint_and_config(args)
    if not 0 < args.alpha <= 1:
        raise ValueError("--alpha must be in (0, 1].")
    num_classes = int(cfg["model"].get("num_classes", 5))
    if not 0 <= args.class_id < num_classes:
        raise ValueError(f"--class-id must be between 0 and {num_classes-1}.")
    row, image_path, count = select_sample(
        cfg, split=args.split, class_id=args.class_id, sample_index=args.sample_index, image_id=args.image_id)
    device = resolve_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment. Use --device cpu.")
    m = cfg["model"]
    model = HybridOrdinalNet(
        backbone=m.get("backbone", "efficientnet_v2_s"), pretrained_imagenet=False,
        projection_hidden_dim=int(m.get("projection_hidden_dim", 1280)), projection_dim=int(m.get("projection_dim", 128)),
        regression_hidden_dim=int(m.get("regression_hidden_dim", 1280)), dropout=float(m.get("dropout", 0.0)),
        regression_input=m.get("regression_input", "backbone"), spatial_attention=True,
        spatial_attention_kernel_size=int(m.get("spatial_attention_kernel_size", 7)), pooling=m.get("pooling", "avg"),
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    with Image.open(image_path) as handle:
        original = ImageOps.exif_transpose(handle).convert("RGB")
    size = int(cfg["data"].get("image_size", 384))
    image = make_eval_transform(size, cfg["data"].get("normalize", "none"))(original)
    view = make_eval_transform(size, "none")(original)
    captured = {}

    def capture_gate(_module, _inputs, logits):
        captured["gate"] = logits.detach().sigmoid().float().cpu()

    hook = model.spatial_attention.conv.register_forward_hook(capture_gate)
    try:
        with torch.inference_mode():
            outputs = model(image.unsqueeze(0).to(device))
    finally:
        hook.remove()
    gate = captured["gate"]
    if gate.ndim != 4 or gate.shape[:2] != (1, 1) or not torch.isfinite(gate).all():
        raise RuntimeError("Unexpected or non-finite Spatial Attention output.")
    gate = gate[0, 0]
    prediction = int(ordinal_class_predictions(outputs["prediction"], num_classes).item())
    score = float(outputs["prediction"].item())
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    label, image_id = int(row["_label"]), str(row["_image_id"])
    display = save_visuals(output_dir, original, view, gate, image_id=image_id,
                           label=label, prediction=prediction, score=score, alpha=args.alpha)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b""):
            digest.update(block)
    report = {
        "checkpoint": str(path), "checkpoint_sha256": digest.hexdigest(), "checkpoint_epoch": checkpoint.get("epoch"),
        "image_id": image_id, "image_path": str(image_path), "subject_id": str(row["_subject_id"]),
        "ground_truth_class": label, "predicted_class": prediction, "regression_prediction": score,
        "correct": prediction == label, "split": args.split, "eligible_class_samples": count,
        "selection": "explicit image id" if args.image_id is not None else f"image-id order, index {args.sample_index}",
        "split_config": cfg.get("split", {}), "seed": int(cfg.get("seed", 42)), "model_config": cfg["model"],
        "input_size": size, "normalize": cfg["data"].get("normalize", "none"), "device": str(device),
        "labels_csv": cfg["data"]["labels_csv"], "image_root": cfg["data"]["image_root"],
        "map": "sigmoid output of the trained spatial_attention.conv, applied by the model in this forward pass",
        "native_map_shape": list(gate.shape), "map_min": float(gate.min()), "map_max": float(gate.max()),
        "map_mean": float(gate.mean()), "map_std": float(gate.std(unbiased=False)), **display,
        "outputs": ["input.png", "spatial_attention.npy", "spatial_attention_input.npy", "spatial_attention.png",
                    "overlay_input.png", "overlay_original.png", "comparison.png", "comparison.pdf"],
    }
    save_json(output_dir / "metadata.json", report)
    print(f"image={image_id} | true_class={label} | predicted_class={prediction} | score={score:.6f}")
    print(f"native_map={list(gate.shape)} | min={gate.min():.6f} | mean={gate.mean():.6f} | max={gate.max():.6f}")
    print(f"Saved: {output_dir / 'comparison.png'}")
    print(f"Original-resolution overlay: {output_dir / 'overlay_original.png'}")
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", help="Spatial Attention run directory containing best.pt.")
    source.add_argument("--checkpoint", help="Exact path to a Spatial Attention checkpoint.")
    parser.add_argument("--config", help="Override/fallback training config; saved checkpoint config is the default.")
    parser.add_argument("--class-id", type=int, default=3, help="Ground-truth class, default: 3.")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    sample = parser.add_mutually_exclusive_group()
    sample.add_argument("--sample-index", type=int, default=0, help="Index among sorted images of the requested true class.")
    sample.add_argument("--image-id", help="Exact labelled image id within the selected split and class.")
    parser.add_argument("--labels-csv", help="Override data.labels_csv after relocating the dataset.")
    parser.add_argument("--image-root", help="Override data.image_root after relocating the dataset.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--output-dir", default="outputs/spatial_attention_class3")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
