from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch


def ordinal_class_predictions(predictions: torch.Tensor, num_classes: int) -> torch.Tensor:
    return predictions.round().clamp(0, num_classes - 1).long()


def batch_metrics(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int) -> dict[str, float]:
    classes = ordinal_class_predictions(predictions.detach(), num_classes)
    targets = targets.long()
    accuracy = (classes == targets).float().mean().item()
    mae = torch.abs(classes.float() - targets.float()).mean().item()
    continuous_mae = torch.abs(predictions.detach().float() - targets.float()).mean().item()
    return {"accuracy": accuracy, "mae": mae, "continuous_mae": continuous_mae}


def aggregate_predictions(
    predictions: list[float],
    targets: list[int],
    num_classes: int,
) -> dict[str, object]:
    pred_array = np.asarray(predictions, dtype=np.float32)
    target_array = np.asarray(targets, dtype=np.int64)
    class_preds = np.rint(pred_array).clip(0, num_classes - 1).astype(np.int64)

    metrics: dict[str, object] = {
        "accuracy": float((class_preds == target_array).mean()),
        "mae": float(np.abs(class_preds - target_array).mean()),
        "continuous_mae": float(np.abs(pred_array - target_array).mean()),
    }

    per_class = defaultdict(dict)
    for label in range(num_classes):
        mask = target_array == label
        if not mask.any():
            per_class[str(label)] = {"support": 0, "accuracy": None, "mae": None}
            continue
        per_class[str(label)] = {
            "support": int(mask.sum()),
            "accuracy": float((class_preds[mask] == target_array[mask]).mean()),
            "mae": float(np.abs(class_preds[mask] - target_array[mask]).mean()),
        }
    metrics["per_class"] = dict(per_class)
    return metrics
