from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import export_spatial_lesion_boxes as boxes
import run_spatial_and_idrid_best as pipeline
from eyepacs_hybrid_ordinal.idrid import prepare_idrid
from eyepacs_hybrid_ordinal.models import HybridOrdinalNet
from smoke_test_idrid import write_synthetic_dataset


@pytest.fixture
def spatial_checkpoint(tmp_path):
    torch.set_num_threads(1)
    cfg = {"seed": 42, "data": {"image_size": 64, "normalize": "imagenet"},
           "model": {"backbone": "efficientnet_b0", "projection_hidden_dim": 16, "projection_dim": 8,
                     "regression_hidden_dim": 16, "regression_input": "projection_concat", "dropout": 0,
                     "spatial_attention": True, "spatial_attention_kernel_size": 7, "num_classes": 5}}
    model = HybridOrdinalNet(**{k: v for k, v in cfg["model"].items() if k != "num_classes"})
    with torch.no_grad():
        model.spatial_attention.conv.weight.zero_()
        model.regression_head.net[-1].weight.zero_()
        model.regression_head.net[-1].bias.fill_(2)
    path = tmp_path / "spatial.pt"
    torch.save({"model": model.state_dict(), "config": cfg, "epoch": 9}, path)
    return path


def direct_data(tmp_path):
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    image[12:68, 20:80] = [140, 60, 30]
    original = tmp_path / "example.png"
    Image.fromarray(image).save(original)
    mask = np.zeros((80, 100), dtype=np.uint8)
    mask[30:33, 30:33] = 255
    mask[50, 60] = 255
    path = tmp_path / "example_HE.tif"
    Image.fromarray(mask).save(path)
    return original, path


def test_hollow_boxes_align_to_original_pixels_and_leave_interiors_unpainted(tmp_path, spatial_checkpoint):
    image_path, mask_path = direct_data(tmp_path)
    output = tmp_path / "results"
    output.mkdir()
    (output / "metadata.json").write_text(json.dumps({"image_id": "previous_eyepacs_image"}))
    result = boxes.run(boxes.parse_args([
        "--checkpoint", str(spatial_checkpoint), "--image", str(image_path), "--mask", str(mask_path),
        "--true-class", "3", "--device", "cpu", "--output-dir", str(output),
    ]))
    assert result["box_count"] == 2  # The single-pixel lesion is retained.
    assert result["box_fill"] is False
    assert result["ground_truth_class"] == 3 and result["predicted_class"] == 2
    assert result["crop_box_xyxy"] == [12, 4, 88, 76]
    table = pd.read_csv(output / "lesion_boxes.csv")
    assert table[["x0", "y0", "x1", "y1"]].values.tolist() == [[30, 30, 33, 33], [60, 50, 61, 51]]
    assert table.area_pixels.tolist() == [9, 1]
    original = Image.open(image_path)
    marked = Image.open(output / "lesion_boxes_original.png")
    overlay = Image.open(output / "overlay_original.png")
    marked_overlay = Image.open(output / "attention_lesion_boxes_original.png")
    assert marked.size == marked_overlay.size == original.size
    assert marked.getpixel((24, 24)) == marked_overlay.getpixel((24, 24)) == (0, 255, 255)
    assert marked.getpixel((31, 31)) == original.getpixel((31, 31))
    assert marked_overlay.getpixel((31, 31)) == overlay.getpixel((31, 31))
    assert overlay.getpixel((0, 0)) == original.getpixel((0, 0))  # Outside the inference crop.
    np.testing.assert_array_equal(np.load(output / "spatial_attention.npy"), np.full((2, 2), 0.5))
    for filename in result["outputs"]:
        assert (output / filename).stat().st_size > 0
    assert (output / "lesion_box_comparison.pdf").read_bytes().startswith(b"%PDF")
    assert json.loads((output / "lesion_box_metadata.json").read_text())["box_count"] == 2
    assert json.loads((output / "metadata.json").read_text()) == result


def test_masks_are_reoriented_with_the_image_and_not_resized_to_fit(tmp_path):
    image = Image.new("RGB", (12, 8), (30, 40, 50))
    exif = Image.Exif(); exif[274] = 6
    source = tmp_path / "rotated.jpg"; image.save(source, exif=exif)
    binary = np.zeros((8, 12), np.uint8); binary[1:3, 3:6] = 255
    mask = tmp_path / "mask.tif"; Image.fromarray(binary).save(mask)
    loaded = boxes.load_masks(source, [mask])[0]["binary"]
    np.testing.assert_array_equal(loaded, np.rot90(binary > 0, -1))
    Image.new("L", (6, 4)).save(mask)
    with pytest.raises(ValueError, match="differs from original image"):
        boxes.load_masks(source, [mask])


def test_connected_components_keep_tiny_lesions_and_diagonal_connectivity(tmp_path):
    binary = np.zeros((20, 30), dtype=bool)
    binary[1, 1] = binary[2, 2] = True
    binary[10, 15] = True
    masks = [{"type": "MA", "binary": binary, "path": tmp_path / "mask.tif"}]
    result = boxes.component_boxes(masks, padding=0)
    assert [row["area_pixels"] for row in result] == [2, 1]
    assert [result[0][key] for key in ["x0", "y0", "x1", "y1"]] == [1, 1, 3, 3]
    assert len(boxes.component_boxes(masks, min_area=2)) == 1


def idrid_data(tmp_path, archived):
    data = tmp_path / "idrid"
    raw = data / "raw"
    write_synthetic_dataset(raw)
    prepare_idrid(raw, data / "prepared", strict=False)
    grading = raw / "B.%20Disease%20Grading/B. Disease Grading/1. Original Images"
    segmentation = raw / "A.%20Segmentation/A. Segmentation"
    for prefix, split, grading_image in [("a", "Training", grading / "b. Testing Set/IDRiD_007.jpg"),
                                          ("b", "Testing", grading / "a. Training Set/IDRiD_014.jpg")]:
        images = segmentation / f"1. Original Images/{prefix}. {split} Set"
        masks = segmentation / f"2. All Segmentation Groundtruths/{prefix}. {split} Set/2. Haemorrhages"
        images.mkdir(parents=True); masks.mkdir(parents=True)
        # A's IDRiD_01 is NOT grade-0 B's IDRiD_001: use image pixels, not the numeric ID.
        (images / "IDRiD_01.jpg").write_bytes(grading_image.read_bytes())
        binary = np.zeros((48, 64), np.uint8); binary[10:14, 20:25] = 255
        Image.fromarray(binary).save(masks / "IDRiD_01_HE.tif")
    if archived:
        archive = data / "archive/idrid.zip"; archive.parent.mkdir()
        with zipfile.ZipFile(archive, "w") as handle:
            for path in segmentation.rglob("*"):
                if path.is_file():
                    handle.write(path, path.relative_to(raw))
        shutil.rmtree(raw / "A.%20Segmentation")
    return data


@pytest.mark.parametrize("archived", [False, True])
def test_automatic_idrid_selection_matches_pixels_reuses_local_data_and_exports(tmp_path, spatial_checkpoint, archived, monkeypatch):
    data = idrid_data(tmp_path, archived)
    prepared_before = {path.name: path.read_bytes() for path in (data / "prepared").glob("*") if path.is_file()}
    def forbidden(*args, **kwargs):
        pytest.fail("Do not train, prepare grading data again, or download anything.")
    monkeypatch.setattr(boxes.hybrid, "train_one_epoch", forbidden)
    monkeypatch.setattr("eyepacs_hybrid_ordinal.idrid.download_idrid", forbidden)
    monkeypatch.setattr("eyepacs_hybrid_ordinal.idrid.prepare_idrid", forbidden)
    args = boxes.parse_args(["--checkpoint", str(spatial_checkpoint), "--idrid-dir", str(data),
                             "--device", "cpu", "--output-dir", str(tmp_path / "comparison")])
    result = boxes.run(args, allow_synthetic=True)
    assert result["ground_truth_class"] == 3
    assert result["grading_samples"] == ["train:idrid_014"]
    assert result["segmentation_split"] == "test" and result["eligible_class_samples"] == 1
    assert result["box_count"] == 1
    assert "2. All Segmentation Groundtruths/b. Testing Set" in result["masks"][0]["path"]
    assert {path.name: path.read_bytes() for path in (data / "prepared").glob("*") if path.is_file()} == prepared_before
    assert (data / "raw/B.%20Disease%20Grading/B. Disease Grading/1. Original Images/a. Training Set/IDRiD_014.jpg").is_file()


def test_equal_numbering_does_not_substitute_a_different_image_grade(tmp_path):
    data = idrid_data(tmp_path, False)
    args = boxes.parse_args(["--idrid-dir", str(data), "--class-id", "0"])
    with pytest.raises(ValueError, match="Found 0 mask-bearing Class 0"):
        boxes.select_idrid(args, allow_synthetic=True)


@pytest.mark.parametrize("malicious", ["A. Segmentation/../../escaped.txt", "A. Segmentation/link"])
def test_zip_extraction_checks_all_paths_before_writing(tmp_path, malicious):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("A. Segmentation/valid.jpg", b"image")
        member = zipfile.ZipInfo(malicious)
        if malicious.endswith("link"):
            member.external_attr = 0o120777 << 16
        handle.writestr(member, b"bad")
    target = tmp_path / "raw"
    with pytest.raises(ValueError, match="Unsafe segmentation archive path"):
        boxes.extract_segmentation(archive, target)
    assert not target.exists()


def test_pipeline_finishes_box_export_before_starting_idrid(tmp_path, monkeypatch):
    prepared = tmp_path / "data/prepared"; prepared.mkdir(parents=True)
    (prepared / "manifest.json").write_text("{}")
    calls = []
    def export(args):
        assert args.class_id == 3 and args.segmentation_split == "test"
        assert args.idrid_dir == tmp_path / "data"
        assert args.segmentation_root == tmp_path / "segmentation"
        calls.append("boxes")
    def train(source, provenance, **kwargs):
        assert calls == ["boxes"]
        assert source["data"]["image_size"] == 384 and source["model"]["pooling"] == "gem"
        assert not source["model"]["spatial_attention"]
        calls.append("idrid")
    monkeypatch.setattr(pipeline.lesion_boxes, "run", export)
    monkeypatch.setattr(pipeline.comparison, "run_comparison", train)
    pipeline.main(["--lesion-boxes", "--data-dir", str(tmp_path / "data"),
                   "--segmentation-root", str(tmp_path / "segmentation")])
    assert calls == ["boxes", "idrid"]


def test_cli_help_and_missing_mask_validation(capsys):
    with pytest.raises(SystemExit) as help_result:
        boxes.parse_args(["--help"])
    assert help_result.value.code == 0 and "--box-padding" in capsys.readouterr().out
    with pytest.raises(SystemExit) as error:
        boxes.parse_args(["--image", "retina.jpg"])
    assert error.value.code == 2
