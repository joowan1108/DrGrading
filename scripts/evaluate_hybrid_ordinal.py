from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eyepacs_hybrid_ordinal.data import EyePACSDataset, build_split_indices, load_eyepacs_dataframe
from eyepacs_hybrid_ordinal.metrics import aggregate_predictions
from eyepacs_hybrid_ordinal.models import HybridOrdinalNet, load_model_checkpoint
from eyepacs_hybrid_ordinal.splitting import build_nested_split_indices
from eyepacs_hybrid_ordinal.transforms import make_eval_transform
from eyepacs_hybrid_ordinal.utils import load_config, resolve_device, worker_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a hybrid ordinal checkpoint on the configured EyePACS split.")
    parser.add_argument("--config", default="configs/hybrid_eyepacs_efficientnet_v2_s.yaml")
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg.get("device", "auto"))
    data_cfg = cfg["data"]
    split_cfg = cfg["split"]
    num_classes = int(cfg["model"].get("num_classes", 5))

    frame = load_eyepacs_dataframe(
        labels_csv=data_cfg["labels_csv"],
        image_col=data_cfg.get("image_col"),
        label_col=data_cfg.get("label_col"),
        subject_col=data_cfg.get("subject_col"),
        max_samples=data_cfg.get("max_samples"),
    )
    num_folds = int(split_cfg.get("num_folds", 0) or 0)
    if num_folds > 1:
        _, _, evaluation_idx = build_nested_split_indices(
            frame,
            seed=int(cfg.get("seed", 42)),
            val_size=float(split_cfg.get("val_size", 0.2)),
            num_folds=num_folds,
            fold_index=int(split_cfg.get("fold_index", 0)),
            subject_independent=bool(split_cfg.get("subject_independent", True)),
        )
    else:
        _, evaluation_idx = build_split_indices(
            frame,
            seed=int(cfg.get("seed", 42)),
            val_size=float(split_cfg.get("val_size", 0.2)),
            num_folds=0,
            subject_independent=bool(split_cfg.get("subject_independent", True)),
        )
    evaluation_frame = frame.iloc[evaluation_idx].reset_index(drop=True)
    dataset = EyePACSDataset(
        evaluation_frame,
        image_root=data_cfg["image_root"],
        transform=make_eval_transform(
            image_size=int(data_cfg.get("image_size", 300)),
            normalize=data_cfg.get("normalize", "none"),
        ),
        verify_images=bool(data_cfg.get("verify_images", False)),
    )
    workers = worker_count(int(data_cfg.get("num_workers", 0)))
    loader = DataLoader(
        dataset,
        batch_size=int(data_cfg.get("batch_size", 24)),
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )

    model = HybridOrdinalNet(
        backbone=cfg["model"].get("backbone", "efficientnet_v2_s"),
        pretrained_imagenet=False,
        projection_hidden_dim=int(cfg["model"].get("projection_hidden_dim", 1280)),
        projection_dim=int(cfg["model"].get("projection_dim", 128)),
        regression_hidden_dim=int(cfg["model"].get("regression_hidden_dim", 1280)),
        dropout=float(cfg["model"].get("dropout", 0.2)),
        regression_input=cfg["model"].get("regression_input", "backbone"),
    ).to(device)
    checkpoint = load_model_checkpoint(model, args.checkpoint, strict=True)
    model.eval()

    amp_enabled = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    predictions: list[float] = []
    targets_all: list[int] = []
    for images, targets, _ in tqdm(loader, desc="evaluate"):
        images = images.to(device, non_blocking=True)
        with autocast(enabled=amp_enabled):
            outputs = model(images)
        predictions.extend(outputs["prediction"].detach().cpu().float().tolist())
        targets_all.extend(targets.long().tolist())

    metrics = aggregate_predictions(predictions, targets_all, num_classes)
    print(f"checkpoint_epoch={checkpoint.get('epoch', 'unknown')}")
    print(f"accuracy={metrics['accuracy']:.4f}")
    print(f"mae={metrics['mae']:.4f}")
    print(f"continuous_mae={metrics['continuous_mae']:.4f}")
    print(f"rmse={metrics['rmse_loss']:.4f}")
    print(f"correct_rate={metrics['correct_rate']:.4f}")
    print(f"adjacent_rate={metrics['adjacent_rate']:.4f}")
    print(f"non_adjacent_rate={metrics['non_adjacent_rate']:.4f}")
    print(f"underdiagnosis_rate={metrics['underdiagnosis_rate']:.4f}")
    print(f"overdiagnosis_rate={metrics['overdiagnosis_rate']:.4f}")
    print(f"mean_underdiagnosis_distance={metrics['mean_underdiagnosis_distance']:.4f}")
    print(f"mean_overdiagnosis_distance={metrics['mean_overdiagnosis_distance']:.4f}")
    print("per_class=")
    for label, values in metrics["per_class"].items():
        print(f"  {label}: {values}")


if __name__ == "__main__":
    main()
