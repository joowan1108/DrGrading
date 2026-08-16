from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score

from .ordinal_metrics import ordinal_error_distribution


def ordinal_class_predictions(predictions: torch.Tensor, num_classes: int) -> torch.Tensor:
    predictions = torch.nan_to_num(
        predictions.detach().float(),
        nan=0.0,
        posinf=float(num_classes - 1),
        neginf=0.0,
    )
    return predictions.round().clamp(0, num_classes - 1).long()


def batch_metrics(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int) -> dict[str, float]:
    classes = ordinal_class_predictions(predictions.detach(), num_classes)
    targets = targets.long()
    accuracy = (classes == targets).float().mean().item()
    mae = torch.abs(classes.float() - targets.float()).mean().item()
    raw_predictions = predictions.detach().float()
    finite_mask = torch.isfinite(raw_predictions)
    safe_predictions = torch.nan_to_num(
        raw_predictions,
        nan=0.0,
        posinf=float(num_classes - 1),
        neginf=0.0,
    )
    continuous_mae = torch.abs(safe_predictions - targets.float()).mean().item()
    nonfinite_rate = 1.0 - finite_mask.float().mean().item()
    return {"accuracy": accuracy, "mae": mae, "continuous_mae": continuous_mae, "nonfinite_prediction_rate": nonfinite_rate}


def aggregate_predictions(
    predictions: list[float],
    targets: list[int],
    num_classes: int,
) -> dict[str, object]:
    pred_array = np.asarray(predictions, dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.int64)
    finite_mask = np.isfinite(pred_array)
    safe_pred_array = np.nan_to_num(pred_array, nan=0.0, posinf=float(num_classes - 1), neginf=0.0)
    class_preds = np.rint(safe_pred_array).clip(0, num_classes - 1).astype(np.int64)

    supported_classes = np.unique(target_array)
    macro_accuracy = np.mean(
        [
            (class_preds[target_array == label] == label).mean()
            for label in supported_classes
        ]
    )
    quadratic_weighted_kappa = float(
        cohen_kappa_score(
            target_array,
            class_preds,
            labels=list(range(num_classes)),
            weights="quadratic",
        )
    )
    if not np.isfinite(quadratic_weighted_kappa):
        quadratic_weighted_kappa = float(np.array_equal(class_preds, target_array))

    metrics: dict[str, object] = {
        "accuracy": float((class_preds == target_array).mean()),
        "macro_accuracy": float(macro_accuracy),
        "quadratic_weighted_kappa": quadratic_weighted_kappa,
        "mae": float(np.abs(class_preds - target_array).mean()),
        "continuous_mae": float(np.abs(safe_pred_array - target_array).mean()),
        "rmse_loss": float(np.sqrt(np.mean(np.square(safe_pred_array - target_array)))),
        "nonfinite_predictions": int((~finite_mask).sum()),
    }
    metrics.update(ordinal_error_distribution(class_preds, target_array))

    per_class = defaultdict(dict)
    for label in range(num_classes):
        mask = target_array == label
        if not mask.any():
            per_class[str(label)] = {
                "support": 0,
                "accuracy": None,
                "mae": None,
                "correct_count": 0,
                "adjacent_count": 0,
                "non_adjacent_count": 0,
                "within_one_class_count": 0,
                "correct_rate": None,
                "adjacent_rate": None,
                "non_adjacent_rate": None,
                "within_one_class_rate": None,
            }
            continue
        distribution = ordinal_error_distribution(class_preds[mask], target_array[mask])
        per_class[str(label)] = {
            "support": int(mask.sum()),
            "accuracy": float((class_preds[mask] == target_array[mask]).mean()),
            "mae": float(np.abs(class_preds[mask] - target_array[mask]).mean()),
            **distribution,
        }
    metrics["per_class"] = dict(per_class)
    return metrics
