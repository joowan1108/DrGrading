from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eyepacs_hybrid_ordinal.data import (
    ClassStratifiedBatchSampler,
    EyePACSDataset,
    build_split_indices,
    inverse_frequency_class_weights,
    load_eyepacs_dataframe,
)
from eyepacs_hybrid_ordinal.losses import (
    PrototypeContrastiveOrdinalLoss,
    WeightedSupervisedContrastiveOrdinalLoss,
    rmse_loss,
)
from eyepacs_hybrid_ordinal.metrics import aggregate_predictions, batch_metrics
from eyepacs_hybrid_ordinal.models import HybridOrdinalNet
from eyepacs_hybrid_ordinal.transforms import make_eval_transform, make_train_transform
from eyepacs_hybrid_ordinal.utils import (
    AverageMeter,
    build_optimizer,
    class_count_string,
    ensure_dir,
    load_config,
    resolve_device,
    save_checkpoint,
    save_json,
    set_seed,
    worker_count,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hybrid contrastive ordinal regression on EyePACS.")
    parser.add_argument("--config", default="configs/hybrid_eyepacs_efficientnet_v2_s.yaml")
    return parser.parse_args()


def make_dataloaders(cfg: dict):
    data_cfg = cfg["data"]
    split_cfg = cfg["split"]
    seed = int(cfg.get("seed", 42))
    num_classes = int(cfg["model"].get("num_classes", 5))

    frame = load_eyepacs_dataframe(
        labels_csv=data_cfg["labels_csv"],
        image_col=data_cfg.get("image_col"),
        label_col=data_cfg.get("label_col"),
        subject_col=data_cfg.get("subject_col"),
        max_samples=data_cfg.get("max_samples"),
    )
    train_idx, val_idx = build_split_indices(
        frame,
        seed=seed,
        val_size=float(split_cfg.get("val_size", 0.2)),
        num_folds=int(split_cfg.get("num_folds", 0) or 0),
        fold_index=int(split_cfg.get("fold_index", 0)),
        subject_independent=bool(split_cfg.get("subject_independent", True)),
    )

    train_frame = frame.iloc[train_idx].reset_index(drop=True)
    val_frame = frame.iloc[val_idx].reset_index(drop=True)
    print(f"train samples={len(train_frame)} classes={class_count_string(train_frame['_label'].to_numpy(), num_classes)}")
    print(f"val samples={len(val_frame)} classes={class_count_string(val_frame['_label'].to_numpy(), num_classes)}")

    train_dataset = EyePACSDataset(
        train_frame,
        image_root=data_cfg["image_root"],
        transform=make_train_transform(
            image_size=int(data_cfg.get("image_size", 300)),
            normalize=data_cfg.get("normalize", "none"),
        ),
        verify_images=bool(data_cfg.get("verify_images", False)),
    )
    val_dataset = EyePACSDataset(
        val_frame,
        image_root=data_cfg["image_root"],
        transform=make_eval_transform(
            image_size=int(data_cfg.get("image_size", 300)),
            normalize=data_cfg.get("normalize", "none"),
        ),
        verify_images=bool(data_cfg.get("verify_images", False)),
    )

    workers = worker_count(int(data_cfg.get("num_workers", 0)))
    common_loader_kwargs = {
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
    }

    if bool(data_cfg.get("stratified_batches", True)):
        batch_sampler = ClassStratifiedBatchSampler(
            train_dataset.targets,
            batch_size=int(data_cfg.get("batch_size", 24)),
            seed=seed,
        )
        train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, **common_loader_kwargs)
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(data_cfg.get("batch_size", 24)),
            shuffle=True,
            drop_last=True,
            **common_loader_kwargs,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=int(data_cfg.get("batch_size", 24)),
        shuffle=False,
        **common_loader_kwargs,
    )
    return train_loader, val_loader, train_dataset.targets


def compute_losses(outputs: dict, targets: torch.Tensor, class_weights: torch.Tensor, criteria: dict, cfg: dict) -> dict:
    loss_cfg = cfg["loss"]
    sample_weights = class_weights[targets.long()]
    pcol = criteria["pcol"](outputs["pcol"], targets)
    scol = criteria["scol"](outputs["scol"], targets, sample_weights=sample_weights)
    rmse = rmse_loss(outputs["prediction"], targets.float())
    total = (
        float(loss_cfg.get("alpha", 1.0)) * pcol
        + float(loss_cfg.get("beta", 1.0)) * scol
        + float(loss_cfg.get("rmse_weight", 1.0)) * rmse
    )
    return {"total": total, "pcol": pcol, "scol": scol, "rmse": rmse}


def train_one_epoch(model, loader, optimizer, scaler, device, amp_enabled, criteria, class_weights, cfg, epoch: int):
    model.train()
    meters = {name: AverageMeter() for name in ["total", "pcol", "scol", "rmse", "accuracy", "mae"]}
    progress = tqdm(loader, desc=f"train {epoch}", leave=False)

    for images, targets, _ in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp_enabled):
            outputs = model(images)
            losses = compute_losses(outputs, targets, class_weights, criteria, cfg)

        scaler.scale(losses["total"]).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        metric_values = batch_metrics(outputs["prediction"], targets, int(cfg["model"].get("num_classes", 5)))
        for name, loss in losses.items():
            meters[name].update(loss.item(), batch_size)
        meters["accuracy"].update(metric_values["accuracy"], batch_size)
        meters["mae"].update(metric_values["mae"], batch_size)
        progress.set_postfix(loss=f"{meters['total'].avg:.4f}", acc=f"{meters['accuracy'].avg:.3f}")

    return {name: meter.avg for name, meter in meters.items()}


@torch.no_grad()
def evaluate(model, loader, device, amp_enabled, cfg):
    model.eval()
    rmse_meter = AverageMeter()
    predictions: list[float] = []
    targets_all: list[int] = []

    for images, targets, _ in tqdm(loader, desc="val", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).long()
        with autocast(enabled=amp_enabled):
            outputs = model(images)
            loss = rmse_loss(outputs["prediction"], targets.float())

        rmse_meter.update(loss.item(), images.size(0))
        predictions.extend(outputs["prediction"].detach().cpu().float().tolist())
        targets_all.extend(targets.detach().cpu().long().tolist())

    metrics = aggregate_predictions(predictions, targets_all, int(cfg["model"].get("num_classes", 5)))
    metrics["rmse_loss"] = rmse_meter.avg
    return metrics


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    device = resolve_device(cfg.get("device", "auto"))
    output_dir = ensure_dir(cfg["train"]["output_dir"])
    train_loader, val_loader, train_targets = make_dataloaders(cfg)

    model = HybridOrdinalNet(
        backbone=cfg["model"].get("backbone", "efficientnet_v2_s"),
        pretrained_imagenet=bool(cfg["model"].get("pretrained_imagenet", False)),
        projection_hidden_dim=int(cfg["model"].get("projection_hidden_dim", 1280)),
        projection_dim=int(cfg["model"].get("projection_dim", 128)),
        regression_hidden_dim=int(cfg["model"].get("regression_hidden_dim", 1280)),
        dropout=float(cfg["model"].get("dropout", 0.2)),
    ).to(device)

    loss_cfg = cfg["loss"]
    num_classes = int(cfg["model"].get("num_classes", 5))
    criteria = {
        "pcol": PrototypeContrastiveOrdinalLoss(
            temperature=float(loss_cfg.get("temperature", 0.1)),
            num_classes=num_classes,
            margin_scale=float(loss_cfg.get("ordinal_margin_scale", 1.0)),
            normalize_ordinal_distance=bool(loss_cfg.get("normalize_ordinal_distance", False)),
            reduction=loss_cfg.get("reduction", "sum"),
        ),
        "scol": WeightedSupervisedContrastiveOrdinalLoss(
            temperature=float(loss_cfg.get("temperature", 0.1)),
            num_classes=num_classes,
            margin_scale=float(loss_cfg.get("ordinal_margin_scale", 1.0)),
            normalize_ordinal_distance=bool(loss_cfg.get("normalize_ordinal_distance", False)),
            reduction=loss_cfg.get("reduction", "sum"),
        ),
    }

    class_weights = torch.tensor(
        inverse_frequency_class_weights(train_targets, num_classes),
        dtype=torch.float32,
        device=device,
    )
    print(f"class_weights={class_weights.detach().cpu().numpy().round(4).tolist()}")

    optimizer = build_optimizer(model.parameters(), cfg)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(cfg["train"].get("lr_plateau_factor", 0.2)),
        patience=int(cfg["train"].get("lr_plateau_patience", 5)),
    )

    amp_enabled = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)

    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    except ImportError:
        writer = None

    best_val_loss = float("inf")
    best_val_metrics = {}
    best_epoch = 0
    bad_epochs = 0
    epochs = int(cfg["train"].get("epochs", 75))

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            amp_enabled,
            criteria,
            class_weights,
            cfg,
            epoch,
        )
        val_metrics = evaluate(model, val_loader, device, amp_enabled, cfg)
        scheduler.step(float(val_metrics["rmse_loss"]))

        if writer is not None:
            for key, value in train_metrics.items():
                writer.add_scalar(f"train/{key}", value, epoch)
            for key, value in val_metrics.items():
                if isinstance(value, (float, int)):
                    writer.add_scalar(f"val/{key}", value, epoch)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": cfg,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        save_checkpoint(output_dir / "last.pt", payload)

        improved = float(val_metrics["rmse_loss"]) < best_val_loss
        if improved:
            best_val_loss = float(val_metrics["rmse_loss"])
            best_val_metrics = val_metrics
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(output_dir / "best.pt", payload)
        else:
            bad_epochs += 1

        save_json(
            output_dir / "metrics.json",
            {
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_val_rmse_loss": best_val_loss,
                "best_val": best_val_metrics,
                "train": train_metrics,
                "val": val_metrics,
            },
        )

        print(
            f"epoch {epoch:03d}: "
            f"train_loss={train_metrics['total']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_rmse={val_metrics['rmse_loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_mae={val_metrics['mae']:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        patience = int(cfg["train"].get("early_stopping_patience", 13))
        if bad_epochs >= patience:
            print(f"early stopping after {bad_epochs} epochs without validation improvement")
            break

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
