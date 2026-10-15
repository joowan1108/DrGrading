"""Prepare the official IDRiD disease-grading split without mixing task folders."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import warnings
import zipfile
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote

DATASET = "aaryapatel98/indian-diabetic-retinopathy-image-dataset"
OFFICIAL_COUNTS = {"train": [134, 20, 136, 74, 49], "test": [34, 5, 32, 19, 13]}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", unquote(value).lower())


def is_grading_path(path: Path) -> bool:
    return any("diseasegrading" in normalized(part) for part in path.parts)


def split_of(path: Path) -> str | None:
    parts = [normalized(part) for part in path.parts]
    if any("training" in part for part in parts):
        return "train"
    if any("testing" in part for part in parts):
        return "test"
    return None


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def audit_duplicates(rows_by_split: dict) -> dict:
    """Describe exact byte duplicates, including their split-specific paths/labels."""
    by_hash = {}
    for split, rows in rows_by_split.items():
        groups = defaultdict(list)
        for row in rows:
            groups[str(row["sha256"])].append({
                "sample_id": str(row["sample_id"]),
                "image": str(row["image"]), "level": int(row["level"]),
            })
        by_hash[split] = groups
    shared = sorted(set(by_hash["train"]) & set(by_hash["test"]))
    cross_split = [{
        "sha256": sha,
        "train": by_hash["train"][sha], "test": by_hash["test"][sha],
        "label_conflict": len({r["level"] for s in ("train", "test") for r in by_hash[s][sha]}) > 1,
    } for sha in shared]
    within_split = {
        split: [{"sha256": sha, "samples": samples} for sha, samples in sorted(groups.items()) if len(samples) > 1]
        for split, groups in by_hash.items()
    }
    summary = {
        "cross_split_hash_groups": len(shared),
        "overlapping_train_samples": sum(len(by_hash["train"][sha]) for sha in shared),
        "overlapping_test_samples": sum(len(by_hash["test"][sha]) for sha in shared),
        "cross_split_label_conflict_groups": sum(group["label_conflict"] for group in cross_split),
        "within_train_hash_groups": len(within_split["train"]),
        "within_test_hash_groups": len(within_split["test"]),
    }
    return {"method": "sha256_file_bytes", "summary": summary, "cross_split": cross_split, "within_split": within_split}


def extract_grading_archive(archive: Path, destination: Path) -> None:
    """Extract only task B; validate all selected paths before writing anything."""
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        selected = []
        for member in handle.infolist():
            relative = Path(member.filename.replace("\\", "/"))
            if not is_grading_path(relative) or member.is_dir():
                continue
            target = (destination / relative).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"Unsafe archive path: {member.filename}")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Symbolic links are not supported in the dataset archive.")
            selected.append((member, target))
        if not selected:
            raise ValueError("The archive does not contain B. Disease Grading.")
        for member, target in selected:
            target.parent.mkdir(parents=True, exist_ok=True)
            with handle.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def download_idrid(data_dir: Path) -> Path:
    """Download on the machine explicitly executing prepare_idrid.py --download."""
    raw = data_dir.resolve() / "raw"
    completion_marker = raw / ".grading-extracted"
    if completion_marker.is_file():
        return raw
    archive_dir = data_dir.resolve() / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive = archive_dir / "indian-diabetic-retinopathy-image-dataset.zip"
    if not archive.exists():
        cli = shutil.which("kaggle")
        command = [cli] if cli else [sys.executable, "-m", "kaggle"]
        try:
            subprocess.run(
                command + ["datasets", "download", "-d", DATASET, "-p", str(archive_dir)],
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise RuntimeError(
                "Kaggle download failed. Install kaggle in this Python environment and "
                "configure Kaggle authentication on the remote computer; then retry."
            ) from error
    if not zipfile.is_zipfile(archive):
        raise ValueError(f"Incomplete/invalid ZIP: {archive}. Move it aside and retry the download.")
    extract_grading_archive(archive, raw)
    completion_marker.write_text(DATASET + "\n", encoding="utf-8")
    return raw


def prepare_idrid(source_dir: Path, output_dir: Path, *, strict: bool = True, overlap_policy: str = "error") -> dict:
    if overlap_policy not in {"error", "report"}:
        raise ValueError("overlap_policy must be 'error' or 'report'.")
    source_dir, output_dir = source_dir.resolve(), output_dir.resolve()
    csvs: dict[str, list[Path]] = {"train": [], "test": []}
    images: dict[tuple[str, str], list[Path]] = {}
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file() or not (is_grading_path(source_dir) or is_grading_path(path.relative_to(source_dir))):
            continue
        split = split_of(path.relative_to(source_dir))
        if split is None:
            continue
        if path.suffix.lower() == ".csv":
            csvs[split].append(path)
        elif path.suffix.lower() in IMAGE_SUFFIXES:
            images.setdefault((split, path.stem.lower()), []).append(path)

    rows_by_split = {}
    provenance = {}
    for split in ("train", "test"):
        if len(csvs[split]) != 1:
            raise ValueError(f"Expected one disease-grading {split} CSV; found {csvs[split]}.")
        label_file = csvs[split][0]
        with label_file.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            headers = {normalized(c): c for c in (reader.fieldnames or [])}
            image_column = headers.get("imagename")
            grade_column = headers.get("retinopathygrade")
            if image_column is None or grade_column is None:
                raise ValueError(f"Missing Image name/Retinopathy grade columns in {label_file}.")
            rows, seen = [], set()
            for record in reader:
                if not any(str(v).strip() for v in record.values() if v is not None):
                    continue
                image_name = str(record[image_column]).strip()
                image_id = Path(image_name).stem.lower()
                if image_id in seen:
                    raise ValueError(f"Duplicate image ID within {split}: {image_name}")
                seen.add(image_id)
                grade = float(record[grade_column])
                if not grade.is_integer() or not 0 <= grade <= 4:
                    raise ValueError(f"Invalid DR grade for {image_name}: {grade}")
                matches = images.get((split, image_id), [])
                if len(matches) != 1:
                    raise ValueError(f"Expected one {split} image for {image_name}; found {matches}.")
                path = matches[0]
                rows.append({
                    "image": path.relative_to(source_dir).as_posix(),
                    "level": int(grade),
                    "sample_id": f"{split}:{image_id}",
                    "sha256": digest(path),
                })
        counts = [sum(row["level"] == grade for row in rows) for grade in range(5)]
        if not rows or (strict and counts != OFFICIAL_COUNTS[split]):
            raise ValueError(f"Unexpected {split} class counts {counts}; expected {OFFICIAL_COUNTS[split]}.")
        rows_by_split[split] = rows
        provenance[split] = {
            "samples": len(rows), "class_counts": counts,
            "labels_source": str(label_file), "labels_sha256": digest(label_file),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_duplicates(rows_by_split)
    audit["overlap_policy"] = overlap_policy
    audit_path = output_dir / "duplicate_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    overlap_count = audit["summary"]["overlapping_test_samples"]
    if overlap_count:
        message = (
            f"Identical image contents: {overlap_count} test images also occur in training. "
            f"Image paths, labels and hashes: {audit_path}. "
        )
        if overlap_policy == "error":
            raise ValueError(
                message + "To preserve the supplied split with disclosed overlap, rerun "
                "prepare_idrid.py with --overlap-policy report. Both full-test and "
                "train-overlap-excluded test metrics will then be saved."
            )
        warnings.warn(message + "Preserving the supplied split; full-test metrics include training duplicates.", UserWarning)
    for split, rows in rows_by_split.items():
        with (output_dir / f"{split}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image", "level", "sample_id", "sha256"])
            writer.writeheader()
            writer.writerows(rows)
    manifest = {
        "dataset": DATASET, "task": "disease_grading", "image_root": str(source_dir),
        "official_split_verified": strict, "splits": provenance,
        "verification_scope": "Reference split/class counts; original dataset identity is not independently verified.",
        "label": "Retinopathy grade (0..4); NOT Risk of macular edema",
        "patient_ids_available": False,
        "overlap_policy": overlap_policy, "duplicate_audit_file": audit_path.name,
        "duplicate_summary": audit["summary"],
        "prepared_csv_sha256": {split: digest(output_dir / f"{split}.csv") for split in rows_by_split},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
