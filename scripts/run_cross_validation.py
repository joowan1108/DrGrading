from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eyepacs_hybrid_ordinal.utils import ensure_dir, load_config, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the paper's subject-independent k-fold EyePACS evaluation.")
    parser.add_argument("--config", default="configs/hybrid_eyepacs_efficientnet_v2_s.yaml")
    return parser.parse_args()


def mean_std(values: list[float]) -> dict[str, float]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=0))}


def available_mean_std(values: list[float | None]) -> dict[str, float | None]:
    available = [float(value) for value in values if value is not None]
    if not available:
        return {"mean": None, "std": None}
    return mean_std(available)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    num_folds = int(cfg["split"].get("num_folds", 10))
    if num_folds < 2:
        raise ValueError("Cross-validation requires split.num_folds >= 2.")

    base_output_dir = ensure_dir(cfg["train"]["output_dir"])
    fold_results = []

    for fold_index in range(num_folds):
        fold_cfg = copy.deepcopy(cfg)
        fold_cfg["split"]["fold_index"] = fold_index
        fold_cfg["train"]["output_dir"] = str(base_output_dir / f"fold_{fold_index}")

        fold_config_path = base_output_dir / f"fold_{fold_index}.yaml"
        with open(fold_config_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(fold_cfg, handle, sort_keys=False)

        command = [
            sys.executable,
            str(ROOT / "scripts" / "train_hybrid_ordinal.py"),
            "--config",
            str(fold_config_path),
        ]
        print(f"running fold {fold_index + 1}/{num_folds}: {' '.join(command)}")
        subprocess.run(command, cwd=str(ROOT), check=True)

        metrics_path = Path(fold_cfg["train"]["output_dir"]) / "metrics.json"
        fold_metrics = load_config(metrics_path)
        test_metrics = fold_metrics.get("test")
        if not test_metrics:
            raise RuntimeError(f"Fold {fold_index} did not produce outer test metrics.")
        best_margins = fold_metrics.get("best_margins", {})
        fold_results.append(
            {
                "fold": fold_index,
                "best_epoch": fold_metrics["best_epoch"],
                "accuracy": float(test_metrics["accuracy"]),
                "mae": float(test_metrics["mae"]),
                "continuous_mae": float(test_metrics["continuous_mae"]),
                "rmse_loss": float(test_metrics["rmse_loss"]),
                "correct_count": int(test_metrics["correct_count"]),
                "adjacent_count": int(test_metrics["adjacent_count"]),
                "non_adjacent_count": int(test_metrics["non_adjacent_count"]),
                "within_one_class_count": int(
                    test_metrics["within_one_class_count"]
                ),
                "correct_rate": float(test_metrics["correct_rate"]),
                "adjacent_rate": float(test_metrics["adjacent_rate"]),
                "non_adjacent_rate": float(test_metrics["non_adjacent_rate"]),
                "within_one_class_rate": float(test_metrics["within_one_class_rate"]),
                "per_class": test_metrics["per_class"],
                "learned_margins": best_margins.get("by_boundary", {}),
            }
        )

    class_metric_names = [
        "support",
        "accuracy",
        "mae",
        "correct_rate",
        "adjacent_rate",
        "non_adjacent_rate",
        "within_one_class_rate",
    ]
    per_class = {}
    for label in range(int(cfg["model"].get("num_classes", 5))):
        label_key = str(label)
        per_class[label_key] = {
            metric: available_mean_std(
                [item["per_class"][label_key][metric] for item in fold_results]
            )
            for metric in class_metric_names
        }

    aggregate = {
        "folds": fold_results,
        "total_counts": {
            metric: sum(int(item[metric]) for item in fold_results)
            for metric in [
                "correct_count",
                "adjacent_count",
                "non_adjacent_count",
                "within_one_class_count",
            ]
        },
        "accuracy": mean_std([item["accuracy"] for item in fold_results]),
        "mae": mean_std([item["mae"] for item in fold_results]),
        "continuous_mae": mean_std([item["continuous_mae"] for item in fold_results]),
        "rmse_loss": mean_std([item["rmse_loss"] for item in fold_results]),
        "correct_rate": mean_std([item["correct_rate"] for item in fold_results]),
        "adjacent_rate": mean_std([item["adjacent_rate"] for item in fold_results]),
        "non_adjacent_rate": mean_std(
            [item["non_adjacent_rate"] for item in fold_results]
        ),
        "within_one_class_rate": mean_std(
            [item["within_one_class_rate"] for item in fold_results]
        ),
        "per_class": per_class,
    }
    boundaries = sorted(
        {
            boundary
            for item in fold_results
            for boundary in item["learned_margins"]
        }
    )
    if boundaries:
        aggregate["learned_margins"] = {
            boundary: mean_std(
                [item["learned_margins"][boundary] for item in fold_results]
            )
            for boundary in boundaries
        }
    save_json(base_output_dir / "cross_validation_metrics.json", aggregate)
    print("cross-validation summary")
    for metric in [
        "accuracy",
        "mae",
        "continuous_mae",
        "rmse_loss",
        "adjacent_rate",
        "non_adjacent_rate",
        "within_one_class_rate",
    ]:
        values = aggregate[metric]
        print(f"{metric}: {values['mean']:.4f} +/- {values['std']:.4f}")
    for boundary, values in aggregate.get("learned_margins", {}).items():
        print(f"margin {boundary}: {values['mean']:.4f} +/- {values['std']:.4f}")


if __name__ == "__main__":
    main()
