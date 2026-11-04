"""Export the actual CBAM spatial gate for one labelled EyePACS image."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import shlex
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPATIAL_RUN = "hybrid_eyepacs_efficientnet_v2_s_softplus_squared_mmnp_spatial_cbam"
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
        f"folder, e.g. {DEFAULT_SPATIAL_RUN}, or pass --checkpoint /actual/path/best.pt. "
        "Find actual checkpoints with: python scripts/export_spatial_attention.py --list-runs"
    )


def saved_config(checkpoint, folder):
    cfg = checkpoint.get("config")
    if not isinstance(cfg, dict):
        recorded = folder / "run_config.json"
        cfg = json.loads(recorded.read_text())["config"] if recorded.is_file() else None
    return cfg


def validation_metrics(checkpoint, folder):
    """Read validation at best.pt, never the final epoch's train/test/val metrics."""
    if isinstance(checkpoint.get("val_metrics"), dict) and checkpoint["val_metrics"]:
        return checkpoint["val_metrics"], "best.pt:val_metrics"
    for filename, key, epoch_key in [("best_metrics.json", "val", "epoch"),
                                      ("metrics.json", "best_val", "best_epoch")]:
        path = folder / filename
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text())
            epoch = checkpoint.get("epoch")
            if epoch is not None and payload.get(epoch_key) != epoch:
                continue
            if isinstance(payload.get(key), dict) and payload[key]:
                return payload[key], f"{filename}:{key}"
        except (OSError, ValueError, AttributeError):
            continue
    return {}, None


def accuracy_percent(metrics):
    if not isinstance(metrics, dict):
        return None
    for key, scale, limit in [("accuracy_percent", 1, 100), ("accuracy", 100, 1)]:
        value = metrics.get(key)
        if value is not None and not isinstance(value, bool):
            try:
                value = float(value)
                if np.isfinite(value) and 0 <= value <= limit:
                    return scale * value
            except (ValueError, TypeError):
                pass
    return None


def list_runs(search_root=None, *, validation_accuracy=False):
    """Inspect actual best.pt weights, rather than trusting experiment folder names."""
    if search_root is None:
        roots = {ROOT / "checkpoints"}
        roots.update(project / "checkpoints" for project in ROOT.parent.iterdir() if project.is_dir())
    else:
        roots = {Path(search_root).expanduser().resolve()}
    folders = set()
    for root in roots:
        if root.is_dir():
            for filename in ["best.pt", "run_config.json"]:
                folders.update(path.parent.resolve() for path in root.rglob(filename))
    records = []
    for folder in sorted(folders):
        path = folder / "best.pt"
        record = {"run_dir": str(folder), "checkpoint": str(path), "best_checkpoint_exists": path.is_file()}
        try:
            checkpoint = {}
            if path.is_file():
                try:
                    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
                except RuntimeError as error:
                    if "mmap" not in str(error):
                        raise
                    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            cfg = saved_config(checkpoint, folder)
            state = clean_state_dict_keys(checkpoint.get("model", checkpoint.get("state_dict", checkpoint)))
            weight = state.get("spatial_attention.conv.weight")
            flag = cfg.get("model", {}).get("spatial_attention", False) if isinstance(cfg, dict) else None
            record.update(spatial_attention_config=flag, has_trained_spatial_weights=isinstance(weight, torch.Tensor),
                          spatial_weight_shape=list(weight.shape) if isinstance(weight, torch.Tensor) else None)
            if validation_accuracy and isinstance(weight, torch.Tensor):
                val, source = validation_metrics(checkpoint, folder)
                record.update(validation_accuracy_percent=accuracy_percent(val), validation_metrics_source=source,
                              validation_class_accuracy_percent={str(k): accuracy_percent(val.get("per_class", {}).get(str(k)))
                                                                 for k in range(5)})
            if not path.is_file():
                record["status"] = "BEST MISSING"
            elif not isinstance(weight, torch.Tensor):
                record["status"] = "NO SPATIAL WEIGHTS"
            elif not isinstance(cfg, dict):
                record["status"] = "CONFIG MISSING"
            elif not flag:
                record["status"] = "DISABLED IN SAVED CONFIG"
            else:
                record["status"] = "READY"
                record["command"] = shlex.join([
                    "python", "scripts/export_spatial_attention.py", "--checkpoint", str(path),
                    "--class-id", "3", "--split", "val", "--device", "cuda",
                    "--output-dir", "outputs/spatial_attention_class3",
                ])
        except (OSError, ValueError, KeyError, RuntimeError, TypeError, pickle.UnpicklingError, EOFError) as error:
            record.update(status="INSPECTION FAILED", error=str(error))
        records.append(record)
        if validation_accuracy:
            continue
        print(f"\n{path}\n  {record['status']} | config.spatial_attention="
              f"{record.get('spatial_attention_config')} | trained_weights={record.get('has_trained_spatial_weights')}")
        if "command" in record:
            print(f"  {record['command']}")
        if "error" in record:
            print(f"  {record['error']}")
    if validation_accuracy:
        records = [row for row in records if row.get("has_trained_spatial_weights")]
        records.sort(key=lambda row: (row.get("validation_accuracy_percent") is None,
                                     -(row.get("validation_accuracy_percent") or 0), row["run_dir"]))
        rows = []
        for record in records:
            path = Path(record["run_dir"])
            run_name = str(path.relative_to(ROOT.parent)) if path.is_relative_to(ROOT.parent) else str(path)
            values = [record.get("validation_accuracy_percent")]
            values += [record.get("validation_class_accuracy_percent", {}).get(str(k)) for k in range(5)]
            rows.append([run_name, *[f"{v:.2f}" if v is not None else "N/A" for v in values]])
        if rows:
            headings = ["Run", "Acc (%)", *[f"C{k} (%)" for k in range(5)]]
            widths = [max(len(row[i]) for row in [headings, *rows]) for i in range(7)]
            for row in [headings, *rows]:
                print("  ".join(value.ljust(width) if i == 0 else value.rjust(width)
                                for i, (value, width) in enumerate(zip(row, widths))))
        else:
            print("No best.pt checkpoints with Spatial Attention weights found. Use --search-root /actual/project/checkpoints.")
    elif not records:
        print("No saved runs found. Set --search-root /actual/project/checkpoints to inspect another location.")
    return records


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
    cfg = load_config(args.config) if args.config else saved_config(checkpoint, path.parent)
    if "spatial_attention.conv.weight" not in state:
        flag = cfg.get("model", {}).get("spatial_attention", False) if isinstance(cfg, dict) else None
        raise ValueError(
            f"This checkpoint has no trained Spatial Attention weights.\nCheckpoint: {path}\n"
            f"Saved config: spatial_attention={flag}. A folder name containing 'spatial' does not enable attention.\n"
            "Find actual trained candidates with: python scripts/export_spatial_attention.py --list-runs\n"
            "IDRiD training can run separately with scripts/compare_idrid_best_eyepacs.py."
        )
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
    source.add_argument("--list-runs", action="store_true", help="List saved settings and actual spatial weights; no inference or training.")
    parser.add_argument("--search-root", type=Path, help="For --list-runs; default: this and sibling projects' checkpoints/.")
    parser.add_argument("--validation-accuracy", action="store_true",
                        help="With --list-runs: show only checkpoints with spatial weights, with validation Acc and C0-C4 in percent.")
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
    args = parser.parse_args(argv)
    if args.search_root is not None and not args.list_runs:
        parser.error("--search-root requires --list-runs.")
    if args.validation_accuracy and not args.list_runs:
        parser.error("--validation-accuracy requires --list-runs.")
    return args


def main(argv=None):
    args = parse_args(argv)
    return list_runs(args.search_root, validation_accuracy=args.validation_accuracy) if args.list_runs else run(args)


if __name__ == "__main__":
    main()
