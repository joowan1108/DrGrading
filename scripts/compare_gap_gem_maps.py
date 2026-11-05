"""Compare saved GAP/GeM regression maps on the exact same original images."""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime
import json
from pathlib import Path
import pickle
import re
import shutil
import sys

import numpy as np
from PIL import Image, ImageOps
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import export_spatial_attention as spatial
import export_spatial_lesion_boxes as lesions
from eyepacs_hybrid_ordinal.idrid import digest, pixel_digest
from eyepacs_hybrid_ordinal.metrics import ordinal_class_predictions
from eyepacs_hybrid_ordinal.models import HybridOrdinalNet, clean_state_dict_keys
from eyepacs_hybrid_ordinal.pooling_visualization import (
    lesion_map_metrics, positive_unit_map, regression_maps, resize_map,
)
from eyepacs_hybrid_ordinal.transforms import make_eval_transform
from eyepacs_hybrid_ordinal.utils import load_config, resolve_device, save_json

GAP_RUN = "hybrid_eyepacs_efficientnet_v2_s_384_avgpool_no_spatial"
GEM_RUN = "hybrid_eyepacs_efficientnet_v2_s_384_gem_no_spatial"


def resolve_checkpoint(value):
    requested = Path(value).expanduser()
    if requested.exists():
        path = requested / "best.pt" if requested.is_dir() else requested
    elif not requested.is_absolute() and (requested.suffix not in {".pt", ".pth"} or requested.name == "best.pt"):
        name = requested.parent.name if requested.name.endswith(".pt") else requested.name
        matches = {p.resolve() for project in ROOT.parent.iterdir() if project.is_dir()
                   for p in [project / "checkpoints" / name / "best.pt"] if p.is_file()}
        local = ROOT / "checkpoints" / name / "best.pt"
        if local.is_file():
            matches.add(local.resolve())
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one existing best.pt for {requested}; found {len(matches)}. "
                                    "Pass its full path with --gap-checkpoint or --gem-checkpoint. "
                                    "Inspect available models with --list-runs.")
        path = matches.pop()
    else:
        path = requested
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}. Use --list-runs to find trained GAP/GeM models.")
    return path.resolve()


def read_checkpoint(path, role=None, config_path=None):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except RuntimeError as error:
        if "mmap" not in str(error):
            raise
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = clean_state_dict_keys(checkpoint.get("model", checkpoint.get("state_dict", checkpoint)))
    cfg = load_config(config_path) if config_path else spatial.saved_config(checkpoint, path.parent)
    if isinstance(cfg, dict) and "model" not in cfg and isinstance(cfg.get("config"), dict):
        cfg = cfg["config"]
    if not isinstance(cfg, dict):
        raise ValueError(f"No saved configuration for {path}. Supply the matching --{role or 'gap'}-config.")
    cfg = copy.deepcopy(cfg)
    model = cfg["model"]
    has_p = "global_pool.p" in state
    pooling = model.get("pooling", "avg")
    actual_role = "gem" if has_p else "gap"
    if pooling not in {"avg", "gem"} or (pooling == "gem") != has_p:
        raise ValueError(f"Pooling configuration and trained GeM weights disagree: {path}")
    if role is not None and actual_role != role:
        raise ValueError(f"Expected a trained {role.upper()} checkpoint, found {actual_role.upper()}: {path}")
    if model.get("spatial_attention", False) or any(k.startswith("spatial_attention.") for k in state):
        raise ValueError(f"Spatial Attention must be disabled for this GAP/GeM comparison: {path}")
    if (model.get("balanced_softmax", False) or float(cfg.get("loss", {}).get("classification_weight", 0)) != 0
            or any(k.startswith("classification_head.") for k in state)):
        raise ValueError(f"Use the non-Balanced-Softmax checkpoints: {path}")
    if not model.get("backbone", "efficientnet_v2_s").lower().startswith("efficientnet"):
        raise ValueError("The comparison requires the EfficientNet convolutional feature path.")
    project = next((p.parent for p in path.parents if p.name == "checkpoints"), ROOT)
    for key in ["labels_csv", "image_root"]:
        if key in cfg.get("data", {}):
            value = Path(cfg["data"][key]).expanduser()
            cfg["data"][key] = str((value if value.is_absolute() else project / value).resolve())
    val, source = spatial.validation_metrics(checkpoint, path.parent)
    metadata = {"path": str(path), "epoch": checkpoint.get("epoch"), "pooling": actual_role,
                "config": cfg, "validation_metrics": val, "validation_metrics_source": source,
                "validation_accuracy_percent": spatial.accuracy_percent(val)}
    return cfg, state, metadata


def pair_differences(gap, gem):
    """Compare the recorded controls, excluding only pooling and output locations."""
    def controls(cfg):
        result = {k: copy.deepcopy(cfg.get(k, {})) for k in ["data", "split", "model", "loss", "train"]}
        result["seed"] = cfg.get("seed", 42)
        for key in ["num_workers", "verify_images"]:
            result["data"].pop(key, None)
        result["model"].pop("pooling", None)
        result["train"].pop("output_dir", None)
        return result
    def flatten(value, prefix=""):
        out = {}
        for key, item in value.items():
            full = f"{prefix}.{key}" if prefix else key
            out.update(flatten(item, full) if isinstance(item, dict) else {full: item})
        return out
    left, right = flatten(controls(gap)), flatten(controls(gem))
    return {k: {"gap": left.get(k), "gem": right.get(k)} for k in sorted(left.keys() | right.keys())
            if left.get(k) != right.get(k)}


def validate_pair(gap, gem, allow_differences=False):
    for key, default in [("image_size", 384), ("normalize", "none")]:
        if gap["data"].get(key, default) != gem["data"].get(key, default):
            raise ValueError(f"The two models require different input {key}; this is not a pooling-only comparison.")
    differences = pair_differences(gap, gem)
    if differences and not allow_differences:
        raise ValueError("Recorded settings differ beyond pooling:\n" + json.dumps(differences, indent=2)
                         + "\nUse a matched pair, or --allow-config-differences for a labelled exploratory comparison.")
    return differences


def build_model(cfg, state, device):
    m = cfg["model"]
    model = HybridOrdinalNet(
        backbone=m.get("backbone", "efficientnet_v2_s"), pretrained_imagenet=False,
        projection_hidden_dim=int(m.get("projection_hidden_dim", 1280)), projection_dim=int(m.get("projection_dim", 128)),
        regression_hidden_dim=int(m.get("regression_hidden_dim", 1280)), dropout=float(m.get("dropout", 0)),
        regression_input=m.get("regression_input", "backbone"), spatial_attention=False,
        pooling=m.get("pooling", "avg"),
    )
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)


def select_cases(args, cfg):
    if args.case_metadata:
        cases = []
        for filename in args.case_metadata:
            filename = filename.expanduser().resolve()
            record = json.loads(filename.read_text())
            image = Path(record["image_path"]).expanduser()
            image = (image if image.is_absolute() else filename.parent / image).resolve()
            if record.get("image_pixel_sha256") and pixel_digest(image) != record["image_pixel_sha256"]:
                raise ValueError(f"Source image pixels no longer match {filename}")
            masks = []
            for entry in record.get("masks", []):
                path = Path(entry["path"] if isinstance(entry, dict) else entry).expanduser()
                path = (path if path.is_absolute() else filename.parent / path).resolve()
                if isinstance(entry, dict) and entry.get("sha256") and digest(path) != entry["sha256"]:
                    raise ValueError(f"Source lesion mask no longer matches {filename}: {path}")
                masks.append(path)
            cases.append({"image_path": str(image), "mask_paths": list(map(str, masks)),
                          "ground_truth_class": record.get("ground_truth_class"),
                          "selection": {"source_metadata": str(filename), "dataset": record.get("dataset", "unspecified"),
                                        "method": "reuse exact previously exported image and masks"}})
        return cases
    if args.image:
        return [{"image_path": str(args.image.expanduser().resolve()),
                 "mask_paths": [str(p.expanduser().resolve()) for p in args.mask or []],
                 "ground_truth_class": args.true_class,
                 "selection": {"dataset": "user-supplied original image", "method": "explicit image path"}}]
    cases = []
    for class_id in args.classes:
        for index in range(args.sample_index, args.sample_index + args.samples_per_class):
            row, path, count = spatial.select_sample(cfg, split=args.split, class_id=class_id, sample_index=index)
            cases.append({"image_path": str(path), "mask_paths": [], "ground_truth_class": int(row["_label"]),
                          "selection": {"dataset": "EyePACS", "split": args.split, "eligible_samples": count,
                                        "method": "fixed image-id order; no prediction or map filtering", "index": index}})
    return cases


def prepare_cases(cases, output, args):
    for index, case in enumerate(cases):
        path = Path(case["image_path"])
        with Image.open(path) as handle:
            original = ImageOps.exif_transpose(handle).convert("RGB")
        masks = lesions.load_masks(path, case["mask_paths"])
        boxes = lesions.component_boxes(masks, min_area=1, padding=args.box_padding)
        crop = spatial.crop_box(original)
        name = re.sub(r"[^\w.-]", "_", path.stem)
        directory = output / f"case_{index+1:02d}_{name}"
        directory.mkdir()
        case.update(image_id=path.stem, directory=str(directory), image_sha256=digest(path),
                    image_pixel_sha256=pixel_digest(path), original_size=list(original.size),
                    crop_box_xyxy=list(crop), boxes=boxes,
                    masks=[{"path": str(m["path"]), "sha256": digest(m["path"]), "type": m["type"]} for m in masks],
                    models={})
        case["_lesion_union"] = np.logical_or.reduce([m["binary"] for m in masks]) if masks else None


def infer_cases(cases, cfg, state, role, device):
    model = build_model(cfg, state, device)
    transform = make_eval_transform(int(cfg["data"].get("image_size", 384)), cfg["data"].get("normalize", "none"))
    try:
        for case in cases:
            with Image.open(case["image_path"]) as handle:
                image = ImageOps.exif_transpose(handle).convert("RGB")
            score, maps = regression_maps(model, transform(image).to(device))
            prediction = int(ordinal_class_predictions(torch.tensor(score), int(cfg["model"].get("num_classes", 5))))
            truth = case["ground_truth_class"]
            record = {"regression_prediction": score, "predicted_class": prediction,
                      "correct": prediction == truth if truth is not None else None,
                      "native_feature_shape_hw": list(maps["gradcam"].shape), "maps": {}}
            if role == "gem":
                record.update(p_raw=float(model.global_pool.p.detach()),
                              p_effective=float(model.global_pool.p.detach().clamp_min(model.global_pool.eps)))
            for name in ["gradcam", "gradient_activation", "moment_gap", "moment_gem"]:
                if name in maps:
                    record["maps"][name] = lesion_map_metrics(maps[name], case["_lesion_union"], tuple(case["crop_box_xyxy"]))
            np.savez_compressed(Path(case["directory"]) / f"{role}_maps.npz", **maps)
            case["models"][role] = record
            print(f"{role.upper()} | {case['image_id']} | true={truth} prediction={prediction} severity={score:.4f}", flush=True)
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def overlay(original, values, crop, alpha, cmap, vmax):
    if not np.isfinite(vmax) or vmax <= 0 or not 0 <= alpha <= 1:
        raise ValueError("Require finite vmax > 0 and alpha in [0,1].")
    display = original.copy()
    display.thumbnail((1600, 1600))
    sx, sy = display.width/original.width, display.height/original.height
    x0, y0, x1, y1 = crop
    box = (round(x0*sx), round(y0*sy), round(x1*sx), round(y1*sy))
    w, h = box[2]-box[0], box[3]-box[1]
    heat = resize_map(values, h, w)
    unit = np.clip(heat/vmax, 0, 1)
    rgb = np.asarray(cmap(unit)[..., :3], dtype=np.float32) * 255
    region = display.crop(box)
    # Opacity tends continuously to zero with attribution; there is no binary mask.
    opacity = (float(alpha) * unit)[..., None]
    background = np.asarray(region, dtype=np.float32)
    blended = np.rint(background*(1-opacity) + rgb*opacity).clip(0, 255).astype(np.uint8)
    display.paste(Image.fromarray(blended), box[:2])
    return display


def render_comparison(cases, output, kind, args, exploratory=False):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle

    stem = f"gap_gem_{kind}"
    pdf_path = output / f"{stem}.pdf"
    pages = []
    with plt.rc_context({"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42}), PdfPages(pdf_path) as pdf:
        for page, first in enumerate(range(0, len(cases), 3), 1):
            batch = cases[first:first+3]
            fig, axes = plt.subplots(len(batch), 3, figsize=(7.2, 2.35*len(batch)+0.6),
                                     squeeze=False, constrained_layout=True)
            for row, case in enumerate(batch):
                with Image.open(case["image_path"]) as handle:
                    original = ImageOps.exif_transpose(handle).convert("RGB")
                with np.load(Path(case["directory"]) / "gem_maps.npz") as saved:
                    if kind == "pooling_moments":
                        maps = [saved["moment_gap"], saved["moment_gem"]]
                    else:
                        maps = [None, saved[kind]]
                if kind != "pooling_moments":
                    with np.load(Path(case["directory"]) / "gap_maps.npz") as saved:
                        maps[0] = saved[kind]
                    maps = [positive_unit_map(m) for m in maps]
                    vmax = 1.0
                    titles = [f"{'GAP' if role == 'gap' else 'GeM'} | pred {case['models'][role]['predicted_class']} | "
                              f"score {case['models'][role]['regression_prediction']:.2f}" for role in ["gap", "gem"]]
                else:
                    vmax = max(float(m.max()) for m in maps)
                    titles = ["GAP p=1 | same features", f"GeM p={case['models']['gem']['p_effective']:.3f} | same features"]
                cmap = plt.get_cmap("turbo")
                display = original.copy(); display.thumbnail((1600, 1600))
                views = [display] + [overlay(original, m, case["crop_box_xyxy"], args.alpha, cmap, vmax) for m in maps]
                truth = case["ground_truth_class"]
                for axis, view, title in zip(axes[row], views, [f"Original | true {truth if truth is not None else '?'}", *titles]):
                    axis.imshow(view, extent=(-0.5, original.width-0.5, original.height-0.5, -0.5))
                    for box in case["boxes"]:
                        axis.add_patch(Rectangle((box["draw_x0"]-0.5, box["draw_y0"]-0.5),
                                                 box["draw_x1"]-box["draw_x0"], box["draw_y1"]-box["draw_y0"],
                                                 fill=False, edgecolor=lesions.BOX_COLOR, linewidth=0.55))
                    axis.set(xlim=(-0.5, original.width-0.5), ylim=(original.height-0.5, -0.5), title=title)
                    axis.axis("off")
                axes[row, 0].text(0.5, -0.035, case["image_id"], transform=axes[row, 0].transAxes,
                                  ha="center", va="top", fontsize=7)
                colorbar = fig.colorbar(ScalarMappable(norm=Normalize(0, vmax), cmap=cmap),
                                       ax=list(axes[row, 1:]), fraction=0.025, pad=0.012)
                colorbar.ax.tick_params(labelsize=6)
                colorbar.set_label("Moment mass per location" if kind == "pooling_moments" else "Positive map / maximum", fontsize=7)
            title = {"gradcam": "GAP vs GeM | regression Grad-CAM",
                     "gradient_activation": "GAP vs GeM | regression gradient x activation",
                     "pooling_moments": "Pooling moments | one fixed GeM feature map"}[kind]
            if exploratory:
                title += " | settings differ"
            note = ("Channel-normalized powered sums; same GeM features. Opacity scales with value."
                    if kind == "pooling_moments" else
                    "Positive: higher severity. Each map normalized by its maximum; opacity follows value.")
            fig.suptitle(title + "\n" + note, fontsize=8)
            image_path = output / f"{stem}_page{page:02d}.png"
            fig.savefig(image_path, dpi=600, bbox_inches="tight", facecolor="white")
            pdf.savefig(fig, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            pages.append(str(image_path))
    return [str(pdf_path), *pages]


def list_runs(search_root=None):
    roots = ([Path(search_root).expanduser().resolve()] if search_root else
             [ROOT / "checkpoints", *[p / "checkpoints" for p in ROOT.parent.iterdir() if p.is_dir()]])
    paths = sorted({p.resolve() for root in roots if root.is_dir() for p in root.rglob("best.pt")})
    records = []
    for path in paths:
        try:
            _, _, info = read_checkpoint(path)
            record = {k: info[k] for k in ["path", "pooling", "epoch", "validation_accuracy_percent"]}
            records.append(record)
            value = record["validation_accuracy_percent"]
            print(f"{record['pooling'].upper():3} | val Acc {value:.2f}% | {path}" if value is not None else
                  f"{record['pooling'].upper():3} | val Acc N/A | {path}")
        except (OSError, ValueError, KeyError, RuntimeError, TypeError, pickle.UnpicklingError, EOFError) as error:
            print(f"SKIP | {path} | {error}")
    if not records:
        print("No usable trained GAP/GeM checkpoints found. Set --search-root to the checkpoints folder.")
    return records


def resolve_replot_metadata(args):
    if args.replot_dir:
        path = args.replot_dir.expanduser().resolve()
        path = path / "comparison_metadata.json" if path.is_dir() else path
    else:
        candidates = []
        for candidate in (ROOT / "outputs").rglob("comparison_metadata.json"):
            try:
                payload = json.loads(candidate.read_text())
                if isinstance(payload.get("cases"), list) and payload["cases"]:
                    candidates.append(candidate)
            except (OSError, ValueError, AttributeError):
                continue
        if not candidates:
            raise FileNotFoundError("No saved comparison_metadata.json under outputs/. "
                                    "Use --replot-dir with the original complete export folder, containing case_*/gap_maps.npz and gem_maps.npz.")
        path = max(candidates, key=lambda p: (p.stat().st_mtime_ns, str(p)))
    if not path.is_file():
        raise FileNotFoundError(f"Saved comparison metadata not found: {path}")
    return path


def replot(args):
    """Redraw cached scalar fields without loading checkpoints or running inference."""
    metadata_path = resolve_replot_metadata(args)
    report = json.loads(metadata_path.read_text())
    if not isinstance(report.get("cases"), list) or not report["cases"]:
        raise ValueError(f"No saved cases in {metadata_path}")
    source_dir = metadata_path.parent
    print(f"Replot saved maps: {source_dir}", flush=True)
    case_inputs = []
    for case in report["cases"]:
        old_directory = Path(case["directory"])
        local = source_dir / old_directory.name
        cache = local if local.is_dir() else old_directory
        image_path = Path(case["image_path"]).expanduser()
        if args.image_root:
            image_path = spatial.resolve_image_path(args.image_root.expanduser(), case["image_id"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Original image not found: {image_path}. "
                                    "Render on the computer holding the original images, or set --image-root.")
        if case.get("image_pixel_sha256") and pixel_digest(image_path) != case["image_pixel_sha256"]:
            raise ValueError(f"Original image pixels have changed: {image_path}")
        maps = {}
        for role in ["gap", "gem"]:
            path = cache / f"{role}_maps.npz"
            if not path.is_file():
                raise FileNotFoundError(f"Cached map not found: {path}. Supply the complete original export folder; PNG files alone are insufficient.")
            required = ["gradcam", "gradient_activation"] + (["moment_gap", "moment_gem"] if role == "gem" else [])
            with np.load(path, allow_pickle=False) as archive:
                for name in required:
                    if name not in archive or archive[name].ndim != 2 or not np.isfinite(archive[name]).all():
                        raise ValueError(f"Missing or invalid native map {name!r} in {path}")
            maps[role] = path
        case_inputs.append((case, image_path.resolve(), cache.name, maps))
    metrics = source_dir / "comparison_metrics.csv"
    if not metrics.is_file():
        raise FileNotFoundError(f"Original numerical results not found: {metrics}")
    output = (args.output_dir or ROOT / "outputs" / f"gap_gem_maps_continuous_{datetime.now():%Y%m%d_%H%M%S_%f}").expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new --output-dir, or omit it for an automatically dated folder: {output}")
    output.mkdir(parents=True)
    cases = []
    for old, image_path, directory_name, maps in case_inputs:
        case = copy.deepcopy(old)
        directory = output / directory_name
        directory.mkdir()
        case.update(directory=str(directory), image_path=str(image_path))
        case["native_map_sha256"] = {}
        for role, source in maps.items():
            target = directory / f"{role}_maps.npz"
            shutil.copy2(source, target)
            case["native_map_sha256"][role] = digest(source)
        save_json(directory / "metadata.json", case)
        cases.append(case)
    methods = ["gradcam", "gradient_activation"] if args.method == "both" else [args.method]
    files = []
    for name in [*methods, "pooling_moments"]:
        files += render_comparison(cases, output, name, args,
                                   exploratory=not report.get("controlled_pooling_comparison", True))
    shutil.copy2(metrics, output / metrics.name)
    report.update(output_dir=str(output), cases=cases, device="CPU; cached rendering only",
                  rendering={"mode": "continuous_alpha", "opacity": "alpha * clipped normalized map", "maximum_alpha": args.alpha},
                  replot_source_metadata=str(metadata_path), replot_source_sha256=digest(metadata_path),
                  native_maps_unchanged=True, training_performed=False, inference_performed=False,
                  outputs=files + [str(output / metrics.name)])
    save_json(output / "comparison_metadata.json", report)
    print(f"Saved continuous-opacity comparison: {output}", flush=True)
    return report


def run(args):
    if args.list_runs:
        return list_runs(args.search_root)
    if args.replot_dir or args.replot_latest:
        return replot(args)
    cfgs, states, sources = {}, {}, {}
    for role in ["gap", "gem"]:
        path = resolve_checkpoint(getattr(args, role + "_checkpoint") or getattr(args, role + "_run"))
        cfgs[role], states[role], sources[role] = read_checkpoint(path, role, getattr(args, role + "_config"))
        for key in ["labels_csv", "image_root"]:
            if getattr(args, key):
                cfgs[role]["data"][key] = str(getattr(args, key).expanduser().resolve())
    differences = validate_pair(cfgs["gap"], cfgs["gem"], args.allow_config_differences)
    cases = select_cases(args, cfgs["gem"])
    output = (args.output_dir or ROOT / "outputs" / f"gap_gem_maps_{datetime.now():%Y%m%d_%H%M%S_%f}").expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new --output-dir, or omit it for an automatically dated folder: {output}")
    device = resolve_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu.")
    output.mkdir(parents=True)
    prepare_cases(cases, output, args)
    for role in ["gap", "gem"]:
        sources[role]["sha256"] = digest(Path(sources[role]["path"]))
        infer_cases(cases, cfgs[role], states.pop(role), role, device)
    methods = ["gradcam", "gradient_activation"] if args.method == "both" else [args.method]
    files = []
    for name in [*methods, "pooling_moments"]:
        files += render_comparison(cases, output, name, args, exploratory=bool(differences))
    rows = []
    for case in cases:
        del case["_lesion_union"]
        for role, prediction in case["models"].items():
            for method, metrics in prediction["maps"].items():
                rows.append({"image_id": case["image_id"], "image_path": case["image_path"],
                             "true_class": case["ground_truth_class"], "model": role, "method": method,
                             "predicted_class": prediction["predicted_class"], "regression_prediction": prediction["regression_prediction"],
                             "correct": prediction["correct"], **metrics})
        save_json(Path(case["directory"]) / "metadata.json", case)
    with (output / "comparison_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    report = {"output_dir": str(output), "device": str(device), "inference_dtype": "float32 without autocast",
              "models": sources, "cases": cases,
              "controlled_pooling_comparison": not bool(differences), "config_differences": differences,
              "target": "raw, unrounded, unclamped scalar regression prediction; positive maps support higher severity",
              "attribution_display": "ReLU, then divide by each map maximum; zero evidence stays zero",
              "moment_definition": "mean_c(x[c,u]^p / sum_v x[c,v]^p), on the same clamped GeM feature map for both p values",
              "lesion_metrics_domain": "original mask pixels inside the inference crop; padded boxes are display-only",
              "interpretation": "Regression maps are sensitivities, not learned spatial gates; pooling moments are not prediction attributions.",
              "rendering": {"mode": "continuous_alpha", "opacity": "alpha * clipped normalized map", "maximum_alpha": args.alpha},
              "training_performed": False, "inference_performed": True,
              "outputs": files + [str(output / "comparison_metrics.csv")]}
    save_json(output / "comparison_metadata.json", report)
    print(f"Saved GAP/GeM comparison: {output}", flush=True)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gap-run", type=Path, default=Path("checkpoints") / GAP_RUN)
    parser.add_argument("--gem-run", type=Path, default=Path("checkpoints") / GEM_RUN)
    for role in ["gap", "gem"]:
        parser.add_argument(f"--{role}-checkpoint", type=Path)
        parser.add_argument(f"--{role}-config", type=Path)
    parser.add_argument("--labels-csv", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--classes", type=int, nargs="+", choices=range(5), default=list(range(5)),
                        help="Ground-truth classes to visualize; defaults to all five classes (0-4).")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--samples-per-class", type=int, default=1)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image", type=Path, help="Explicit original image, without requiring EyePACS CSVs.")
    source.add_argument("--case-metadata", type=Path, nargs="+", help="Reuse exact images/masks from existing lesion_box_metadata.json files.")
    source.add_argument("--replot-dir", type=Path, help="Redraw all cached cases in a saved export folder, without model inference.")
    source.add_argument("--replot-latest", action="store_true", help="Find the latest saved comparison under outputs/ and redraw cached maps.")
    parser.add_argument("--mask", action="append", type=Path, help="True lesion mask for --image; repeat for multiple lesion types.")
    parser.add_argument("--true-class", type=int, choices=range(5))
    parser.add_argument("--box-padding", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=0.4,
                        help="Maximum overlay opacity; actual opacity scales continuously with normalized map value.")
    parser.add_argument("--method", choices=["both", "gradcam", "gradient_activation"], default="both")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, help="New output folder; omitted creates a unique dated folder.")
    parser.add_argument("--allow-config-differences", action="store_true")
    parser.add_argument("--list-runs", action="store_true", help="Inspect usable non-Spatial, non-BS GAP/GeM models and validation accuracy.")
    parser.add_argument("--search-root", type=Path)
    args = parser.parse_args(argv)
    if args.samples_per_class < 1 or args.sample_index < 0 or args.box_padding < 0 or not 0 < args.alpha <= 1:
        parser.error("Require positive samples-per-class, nonnegative sample-index/box-padding, and alpha in (0,1].")
    if (args.mask or args.true_class is not None) and not args.image:
        parser.error("--mask and --true-class apply only to --image.")
    return args


if __name__ == "__main__":
    run(parse_args())
