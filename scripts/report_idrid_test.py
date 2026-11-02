"""Verify saved held-out predictions and report only the FINAL official IDRiD test."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

OFFICIAL_TEST_COUNTS = [34, 5, 32, 19, 13]


def report(output: Path) -> dict:
    path = output / "evaluation_metrics.json"
    if not path.is_file():
        raise ValueError("No final evaluation_metrics.json. Epoch train_accuracy is NOT test accuracy; training/evaluation has not completed.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("official_test_used_for_training") or "seen_test" in payload:
        raise ValueError("This run evaluated TRAIN-SEEN images. It is resubstitution, not held-out test accuracy; rerun train_idrid_fixed_iterations.py from fresh initialization.")
    if payload.get("evaluation_status") != "complete" or not payload.get("evaluation_is_independent"):
        raise ValueError("This run has no completed, audited held-out evaluation. Use the updated train-only runner.")
    split = json.loads((output / "training_split.json").read_text(encoding="utf-8"))
    with (output / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    sample_ids = [row["sample_id"] for row in rows]
    if not rows or len(set(sample_ids)) != len(rows) or set(sample_ids) != set(split["test"]):
        raise ValueError("Predictions do not cover the saved test membership exactly once.")
    if set(sample_ids) & set(split["train"]) or any(not sid.startswith("test:") for sid in sample_ids):
        raise ValueError("Training IDs appear in final test predictions.")
    if set(split["training_pixel_hashes"]) & set(split["test_pixel_hashes"]):
        raise ValueError("Train/test decoded pixels overlap.")
    confusion = [[0] * 5 for _ in range(5)]
    for row in rows:
        label, prediction = int(row["label"]), int(row["prediction_class"])
        if not 0 <= label < 5 or not 0 <= prediction < 5:
            raise ValueError("Invalid predicted/true grade.")
        if row["split"] != "official_test" or row["overlaps_train"].lower() != "false":
            raise ValueError("Predictions include non-test or overlapping images.")
        confusion[label][prediction] += 1
    counts = [sum(row) for row in confusion]
    if not payload.get("synthetic_smoke_test", False) and counts != OFFICIAL_TEST_COUNTS:
        raise ValueError(f"Expected 103 official test samples with counts {OFFICIAL_TEST_COUNTS}; got {counts}.")
    metrics = payload["test"]
    n = len(rows)
    accuracy = sum(confusion[k][k] for k in range(5)) / n
    mae = sum(abs(i - j) * confusion[i][j] for i in range(5) for j in range(5)) / n
    macro = sum(confusion[k][k] / counts[k] for k in range(5) if counts[k]) / sum(c > 0 for c in counts)
    columns = [sum(confusion[i][j] for i in range(5)) for j in range(5)]
    observed = sum((i - j) ** 2 * confusion[i][j] for i in range(5) for j in range(5))
    expected = sum((i - j) ** 2 * counts[i] * columns[j] for i in range(5) for j in range(5)) / n
    qwk = 1 - observed / expected if expected else float(observed == 0)
    for key, value in {"accuracy": accuracy, "mae": mae, "macro_accuracy": macro, "quadratic_weighted_kappa": qwk}.items():
        if not math.isclose(metrics[key], value, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"Saved {key} disagrees with predictions.csv.")
    if metrics["samples"] != n or metrics["confusion_matrix"] != confusion:
        raise ValueError("Saved test sample count/confusion matrix disagrees with predictions.csv.")
    tag = "SYNTHETIC SMOKE TEST" if payload.get("synthetic_smoke_test") else "FINAL OFFICIAL TEST"
    print(f"[{tag}] N={n} | Accuracy={100*accuracy:.2f}% | Macro={100*macro:.2f}% | QWK={qwk:.4f} | MAE={mae:.4f}")
    for grade, count in enumerate(counts):
        value = f"{100 * confusion[grade][grade] / count:.2f}%" if count else "N/A"
        print(f"Class {grade}: {value} ({confusion[grade][grade]}/{count})")
    tsne = payload.get("tsne_correct_test")
    if tsne is not None:
        correct_count = sum(confusion[k][k] for k in range(5))
        if tsne["correct_samples"] != correct_count or tsne["total_test_samples"] != n:
            raise ValueError("t-SNE sample count does not match the correctly classified test subset.")
        with (output / tsne["coordinates_file"]).open(newline="", encoding="utf-8") as handle:
            coordinates = list(csv.DictReader(handle))
        expected_ids = {row["sample_id"] for row in rows if row["label"] == row["prediction_class"]}
        if len(coordinates) != correct_count or {row["sample_id"] for row in coordinates} != expected_ids:
            raise ValueError("t-SNE includes incorrect/non-test samples or omits correct ones.")
        print(f"t-SNE: {tsne['status']}, correct-only n={correct_count}/{n}; {output / tsne['image_file']}")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    try:
        report(args.output_dir)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"NOT A VERIFIED TEST RESULT: {error}\n")


if __name__ == "__main__":
    main()
