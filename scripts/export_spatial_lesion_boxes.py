"""Compare actual lesion-mask boxes with a saved Spatial Attention gate."""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

import export_spatial_attention as spatial
import train_idrid as hybrid
from eyepacs_hybrid_ordinal.idrid import digest, normalized, pixel_digest, split_of
from eyepacs_hybrid_ordinal.utils import resolve_device, save_json

MASK_PATTERN = re.compile(r"^(.*)_(MA|HE|EX|SE)$", re.IGNORECASE)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
BOX_COLOR = "#00FFFF"


def is_segmentation(path):
    return any(normalized(part) in {"asegmentation", "segmentation"} for part in path.parts)


def segmentation_images(root):
    if not root.is_dir():
        return []
    return sorted(path.resolve() for path in root.rglob("*") if path.is_file()
                  and path.suffix.lower() in IMAGE_SUFFIXES and is_segmentation(path)
                  and any("originalimages" in normalized(part) for part in path.parts))


def extract_segmentation(archive, destination):
    """Reuse an existing local ZIP, extracting task A without touching grading data."""
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        selected = []
        for member in handle.infolist():
            relative = Path(member.filename.replace("\\", "/"))
            if member.is_dir() or not is_segmentation(relative):
                continue
            target = (destination / relative).resolve()
            if not target.is_relative_to(destination) or (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Unsafe segmentation archive path: {member.filename}")
            selected.append((member, target))
        if not selected:
            return False
        for member, target in selected:
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with handle.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
    print(f"Reused local ZIP for task A: {archive}", flush=True)
    return True


def find_segmentation(data_dir, manifest, segmentation_root=None):
    if segmentation_root is not None:
        root = Path(segmentation_root).expanduser().resolve()
        images = segmentation_images(root)
        if not images:
            raise FileNotFoundError(f"No A. Segmentation original images under {root}.")
        return root, images
    roots = [Path(manifest["image_root"]), data_dir / "raw", data_dir]
    for root in dict.fromkeys(roots):
        images = segmentation_images(root)
        if images:
            return root, images
    for archive in sorted((data_dir / "archive").glob("*.zip")):
        if extract_segmentation(archive, data_dir / "raw"):
            return data_dir / "raw", segmentation_images(data_dir / "raw")
    raise FileNotFoundError(
        "IDRiD lesion masks were not found. Set --segmentation-root to the existing A. Segmentation "
        "folder, or place the original IDRiD ZIP in DATA_DIR/archive/. No download or grading preparation is performed."
    )


def select_idrid(args, *, allow_synthetic=False):
    data_dir = args.idrid_dir.expanduser().resolve()
    frames, manifest = hybrid.load_prepared(data_dir, allow_synthetic=allow_synthetic)
    root, images = find_segmentation(data_dir, manifest, args.segmentation_root)
    mask_index = defaultdict(list)
    for path in sorted(root.rglob("*")):
        match = MASK_PATTERN.fullmatch(path.stem)
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and is_segmentation(path) and match:
            mask_index[(split_of(path), match[1].lower())].append(path.resolve())
    # Task A numbering need not match task B numbering. Join actual decoded pixels.
    by_pixels = defaultdict(list)
    for frame in frames.values():
        for row in frame.to_dict("records"):
            by_pixels[row["_pixel_sha256"]].append(row)
    candidates = []
    for image in images:
        split = split_of(image)
        if args.segmentation_split != "all" and split != args.segmentation_split:
            continue
        masks = mask_index.get((split, image.stem.lower()), [])
        if not masks:
            continue
        matches = by_pixels.get(pixel_digest(image), [])
        grades = {int(row["_label"]) for row in matches}
        if len(grades) > 1:
            raise ValueError(f"Conflicting grading labels for the actual pixels of {image}.")
        if grades == {args.class_id}:
            candidates.append((image, masks, matches, split))
    if not 0 <= args.sample_index < len(candidates):
        raise ValueError(f"Found {len(candidates)} mask-bearing Class {args.class_id} images in segmentation "
                         f"split {args.segmentation_split}; --sample-index is out of range. Try --segmentation-split all.")
    image, masks, matches, split = candidates[args.sample_index]
    return image, masks, args.class_id, {
        "dataset": "IDRiD segmentation", "segmentation_split": split,
        "grading_match_method": "exact EXIF-oriented RGB pixel SHA256, not filename numbering",
        "grading_samples": [row["sample_id"] for row in matches], "eligible_class_samples": len(candidates),
        "selection": f"image-path order, index {args.sample_index}; no attention-based selection",
    }


def load_masks(image_path, mask_paths):
    with Image.open(image_path) as image:
        raw_size, orientation = image.size, image.getexif().get(274, 1)
    operations = {2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180,
                  4: Image.Transpose.FLIP_TOP_BOTTOM, 5: Image.Transpose.TRANSPOSE,
                  6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE, 8: Image.Transpose.ROTATE_90}
    result = []
    for path in dict.fromkeys(Path(p).expanduser().resolve() for p in mask_paths):
        with Image.open(path) as mask:
            if mask.size != raw_size:
                raise ValueError(f"Mask size {mask.size} differs from original image {raw_size}: {path}. "
                                 "Use the original matching mask; resizing is not automatic.")
            mask = mask.convert("L")
            if orientation in operations:
                mask = mask.transpose(operations[orientation])
            binary = np.asarray(mask) > 0
        match = MASK_PATTERN.fullmatch(path.stem)
        result.append({"path": path, "type": match[2].upper() if match else "lesion", "binary": binary})
    return result


def component_boxes(masks, *, min_area=1, padding=6):
    boxes = []
    for mask in masks:
        labels, count = ndimage.label(mask["binary"], structure=np.ones((3, 3), dtype=np.uint8))
        areas = np.bincount(labels.ravel())
        height, width = labels.shape
        for component, slices in enumerate(ndimage.find_objects(labels, max_label=count), 1):
            if slices is None or areas[component] < min_area:
                continue
            yy, xx = slices
            boxes.append({"lesion_type": mask["type"], "component": component, "mask": str(mask["path"]),
                          "area_pixels": int(areas[component]), "x0": xx.start, "y0": yy.start,
                          "x1": xx.stop, "y1": yy.stop,
                          "draw_x0": max(0, xx.start-padding), "draw_y0": max(0, yy.start-padding),
                          "draw_x1": min(width, xx.stop+padding), "draw_y1": min(height, yy.stop+padding)})
    return boxes


def draw_boxes(image, boxes, line_width):
    result = image.copy()
    draw = ImageDraw.Draw(result)
    for box in boxes:
        side = min(box["draw_x1"]-box["draw_x0"], box["draw_y1"]-box["draw_y0"])
        width = min(line_width, max(1, (side-1)//2))
        draw.rectangle((box["draw_x0"], box["draw_y0"], box["draw_x1"]-1, box["draw_y1"]-1),
                       outline=BOX_COLOR, width=width)
    return result


def save_box_visuals(output, original, boxes, *, image_id, label, prediction, line_width, display_limits):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle

    with Image.open(output / "overlay_original.png") as handle:
        overlay = handle.convert("RGB")
    draw_boxes(original, boxes, line_width).save(output / "lesion_boxes_original.png")
    draw_boxes(overlay, boxes, line_width).save(output / "attention_lesion_boxes_original.png")
    with plt.rc_context({"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), constrained_layout=True)
        for index, (axis, view, title) in enumerate(zip(
                axes, [original, overlay, overlay], ["Lesion boxes", "Spatial Attention", "Attention + boxes"])):
            axis.imshow(view)
            if index != 1:
                for box in boxes:
                    axis.add_patch(Rectangle((box["draw_x0"]-0.5, box["draw_y0"]-0.5),
                                             box["draw_x1"]-box["draw_x0"], box["draw_y1"]-box["draw_y0"],
                                             fill=False, edgecolor=BOX_COLOR, linewidth=0.55))
            axis.set(xlim=(-0.5, original.width-0.5), ylim=(original.height-0.5, -0.5))
            axis.set_title(title)
            axis.axis("off")
        truth = f"Class {label}" if label is not None else "grade not supplied"
        fig.suptitle(f"{image_id} | {truth} | prediction {prediction}", fontsize=8)
        colorbar = fig.colorbar(ScalarMappable(norm=Normalize(*display_limits), cmap="turbo"),
                               ax=list(axes[1:]), fraction=0.025, pad=0.015)
        colorbar.set_label("Spatial weight", fontsize=7)
        colorbar.ax.tick_params(labelsize=6)
        fig.savefig(output / "lesion_box_comparison.png", dpi=600, bbox_inches="tight", facecolor="white")
        fig.savefig(output / "lesion_box_comparison.pdf", bbox_inches="tight", facecolor="white")
        plt.close(fig)


def run(args, *, allow_synthetic=False):
    if not 0 < args.alpha <= 1 or args.min_mask_area < 1 or args.box_padding < 0 or (args.line_width is not None and args.line_width < 1):
        raise ValueError("Require alpha in (0,1], min-mask-area >= 1, box-padding >= 0 and positive line-width.")
    options = ["--checkpoint", str(args.checkpoint)] if args.checkpoint else ["--run-dir", str(args.run_dir)]
    if args.config:
        options += ["--config", str(args.config)]
    checkpoint_path, checkpoint, state, cfg = spatial.checkpoint_and_config(spatial.parse_args(options))
    if args.image is not None:
        image_path, mask_paths, label = args.image.expanduser().resolve(), args.mask, args.true_class
        provenance = {"dataset": "user-supplied image/masks", "grading_match_method": "user supplied"}
    else:
        image_path, mask_paths, label, provenance = select_idrid(args, allow_synthetic=allow_synthetic)
    masks = load_masks(image_path, mask_paths)
    boxes = component_boxes(masks, min_area=args.min_mask_area, padding=args.box_padding)
    if not boxes:
        raise ValueError("No nonempty lesion components remain in the supplied masks. No lesion boxes were fabricated.")
    device = resolve_device(args.device)
    original, view, gate, prediction, score = spatial.infer_image(state, cfg, image_path, device)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    display = spatial.save_visuals(output, original, view, gate, image_id=image_path.stem, label=label,
                                   prediction=prediction, score=score, alpha=args.alpha, input_title="Model input")
    line_width = args.line_width or max(1, round(max(original.size)/1000))
    save_box_visuals(output, original, boxes, image_id=image_path.stem, label=label,
                     prediction=prediction, line_width=line_width, display_limits=display["display_limits"])
    with (output / "lesion_boxes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(boxes[0])); writer.writeheader(); writer.writerows(boxes)
    report = {**provenance, "checkpoint": str(checkpoint_path), "checkpoint_sha256": digest(checkpoint_path),
              "checkpoint_epoch": checkpoint.get("epoch"), "model_config": cfg["model"],
              "input_size": int(cfg["data"]["image_size"]), "normalize": cfg["data"].get("normalize", "none"),
              "image_id": image_path.stem, "image_path": str(image_path), "image_pixel_sha256": pixel_digest(image_path),
              "ground_truth_class": label, "predicted_class": prediction, "regression_prediction": score,
              "correct": prediction == label if label is not None else None,
              "box_color": BOX_COLOR, "box_fill": False, "box_padding_original_pixels": args.box_padding,
              "box_line_width_original_pixels": line_width, "min_mask_area_pixels": args.min_mask_area,
              "box_count": len(boxes), "connectivity": 8, "coordinates": "EXIF-oriented original image; x1/y1 exclusive",
              "masks": [{"path": str(mask["path"]), "sha256": digest(mask["path"]), "type": mask["type"],
                         "positive_pixels": int(mask["binary"].sum())} for mask in masks],
              "boxes": boxes, "spatial_gate_shape": list(gate.shape), **display,
              "interpretation": "Boxes come only from supplied lesion masks; attention is unaltered model output.",
              "outputs": ["lesion_box_comparison.png", "lesion_box_comparison.pdf", "lesion_boxes_original.png",
                          "attention_lesion_boxes_original.png", "lesion_boxes.csv", "spatial_attention.npy"]}
    save_json(output / "lesion_box_metadata.json", report)
    save_json(output / "metadata.json", report)
    print(f"Lesion boxes={len(boxes)} | image={image_path} | true={label} predicted={prediction}", flush=True)
    print(f"Saved: {output / 'lesion_box_comparison.pdf'}", flush=True)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--run-dir", default=f"checkpoints/{spatial.DEFAULT_SPATIAL_RUN}")
    source.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--idrid-dir", type=Path, default=spatial.ROOT / "data/idrid")
    parser.add_argument("--segmentation-root", type=Path)
    parser.add_argument("--segmentation-split", choices=["train", "test", "all"], default="test")
    parser.add_argument("--class-id", type=int, choices=range(5), default=3)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--image", type=Path, help="Explicit original image instead of automatic IDRiD selection.")
    parser.add_argument("--mask", type=Path, action="append", help="Matching lesion mask; repeat for multiple lesion types.")
    parser.add_argument("--true-class", type=int, choices=range(5), help="Optional known grade for --image.")
    parser.add_argument("--min-mask-area", type=int, default=1)
    parser.add_argument("--box-padding", type=int, default=6, help="Display padding in original-image pixels; masks are unchanged.")
    parser.add_argument("--line-width", type=int)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=spatial.ROOT / "outputs/spatial_lesion_boxes_class3")
    args = parser.parse_args(argv)
    if bool(args.image) != bool(args.mask):
        parser.error("--image and one or more --mask paths must be provided together.")
    if args.true_class is not None and args.image is None:
        parser.error("--true-class is used only with --image; IDRiD grades are matched automatically.")
    return args


if __name__ == "__main__":
    run(parse_args())
