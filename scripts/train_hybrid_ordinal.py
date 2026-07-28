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
    load_eyepacs_dataframe,
)
from eyepacs_hybrid_ordinal.losses import (
    CumulativeOrdinalMargins,
    PrototypeContrastiveOrdinalLoss,
    WeightedSupervisedContrastiveOrdinalLoss,
    rmse_loss,
)
from eyepacs_hybrid_ordinal.metrics import aggregate_predictions, batch_metrics
from eyepacs_hybrid_ordinal.models import HybridOrdinalNet, load_model_checkpoint
from eyepacs_hybrid_ordinal.splitting import build_nested_split_indices
from eyepacs_hybrid_ordinal.transforms import make_eval_transform, make_train_transform
from eyepacs_hybrid_ordinal.weighting import batch_inverse_frequency_sample_weights
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


def build_scheduler(optimizer, cfg: dict):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(cfg["train"].get("lr_plateau_factor", 0.2)),
        patience=int(cfg["train"].get("lr_plateau_patience", 5)),
    )


def build_learnable_margins(cfg: dict, device: torch.device):
    loss_cfg = cfg["loss"]
    if not bool(loss_cfg.get("learnable_ordinal_margins", False)):
        return None
    if bool(loss_cfg.get("normalize_ordinal_distance", False)):
        raise ValueError(
            "normalize_ordinal_distance must be false when learnable_ordinal_margins is enabled."
        )

    epochs = int(cfg["train"].get("epochs", 75))
    phase1_epochs = int(loss_cfg.get("margin_phase1_epochs", 10))
    minimum_margin = float(loss_cfg.get("margin_min", 0.05))
    init_min = float(loss_cfg.get("margin_init_min", 0.5))
    collapse_guard = float(loss_cfg.get("margin_collapse_guard", 0.1))
    accuracy_threshold = float(
        loss_cfg.get("margin_phase1_accuracy_threshold", 0.95)
    )
    if not 1 <= phase1_epochs < epochs:
        raise ValueError("margin_phase1_epochs must be between 1 and train.epochs - 1.")
    if not minimum_margin < collapse_guard < init_min:
        raise ValueError("Expected margin_min < margin_collapse_guard < margin_init_min.")
    if not 0.0 <= accuracy_threshold <= 1.0:
        raise ValueError("margin_phase1_accuracy_threshold must be in [0, 1].")

    return CumulativeOrdinalMargins(
        num_classes=int(cfg["model"].get("num_classes", 5)),
        minimum_margin=minimum_margin,
        init_min=init_min,
        init_max=float(loss_cfg.get("margin_init_max", 1.0)),
    ).to(device)


def margin_snapshot(
    learnable_margins: CumulativeOrdinalMargins | None,
    phase: str,
    freeze_reason: str | None,
) -> dict:
    if learnable_margins is None:
        return {"enabled": False, "phase": "fixed"}

    values = learnable_margins.margin_values().detach().cpu().float().tolist()
    return {
        "enabled": True,
        "phase": phase,
        "frozen": not learnable_margins.raw_margins.requires_grad,
        "freeze_reason": freeze_reason,
        "values": values,
        "by_boundary": {
            f"{index}-{index + 1}": value for index, value in enumerate(values)
        },
        "minimum": min(values),
        "maximum": max(values),
    }


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
    num_folds = int(split_cfg.get("num_folds", 0) or 0)
    if num_folds > 1:
        train_idx, val_idx, test_idx = build_nested_split_indices(
            frame,
            seed=seed,
            val_size=float(split_cfg.get("val_size", 0.2)),
            num_folds=num_folds,
            fold_index=int(split_cfg.get("fold_index", 0)),
            subject_independent=bool(split_cfg.get("subject_independent", True)),
        )
    else:
        train_idx, val_idx = build_split_indices(
            frame,
            seed=seed,
            val_size=float(split_cfg.get("val_size", 0.2)),
            num_folds=0,
            subject_independent=bool(split_cfg.get("subject_independent", True)),
        )
        test_idx = None

    train_frame = frame.iloc[train_idx].reset_index(drop=True)
    val_frame = frame.iloc[val_idx].reset_index(drop=True)
    test_frame = frame.iloc[test_idx].reset_index(drop=True) if test_idx is not None else None
    print(f"train samples={len(train_frame)} classes={class_count_string(train_frame['_label'].to_numpy(), num_classes)}")
    print(f"val samples={len(val_frame)} classes={class_count_string(val_frame['_label'].to_numpy(), num_classes)}")
    if test_frame is not None:
        print(f"test samples={len(test_frame)} classes={class_count_string(test_frame['_label'].to_numpy(), num_classes)}")

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
    test_dataset = None
    if test_frame is not None:
        test_dataset = EyePACSDataset(
            test_frame,
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

    batch_size = int(data_cfg.get("batch_size", 24))
    if bool(data_cfg.get("stratified_batches", True)):
        batch_sampler = ClassStratifiedBatchSampler(
            train_dataset.targets,
            batch_size=batch_size,
            seed=seed,
            min_samples_per_class=int(data_cfg.get("stratified_min_samples_per_class", 2)),
        )
        contrastive_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, **common_loader_kwargs)
    else:
        contrastive_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            **common_loader_kwargs,
        )

    if bool(data_cfg.get("regression_natural_batches", True)):
        regression_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            **common_loader_kwargs,
        )
    else:
        regression_loader = contrastive_loader

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **common_loader_kwargs,
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            **common_loader_kwargs,
        )
    return contrastive_loader, regression_loader, val_loader, test_loader


def compute_losses(
    contrastive_outputs: dict,
    contrastive_targets: torch.Tensor,
    regression_outputs: dict,
    regression_targets: torch.Tensor,
    contrastive_sample_weights: torch.Tensor,
    criteria: dict,
    learnable_margins: CumulativeOrdinalMargins | None,
    cfg: dict,
) -> dict:
    loss_cfg = cfg["loss"]
    pcol = criteria["pcol"](
        contrastive_outputs["pcol"],
        contrastive_targets,
        learnable_margins=learnable_margins,
    )
    scol = criteria["scol"](
        contrastive_outputs["scol"],
        contrastive_targets,
        sample_weights=contrastive_sample_weights,
        learnable_margins=learnable_margins,
    )
    rmse = rmse_loss(regression_outputs["prediction"], regression_targets.float())
    total = (
        float(loss_cfg.get("alpha", 1.0)) * pcol
        + float(loss_cfg.get("beta", 1.0)) * scol
        + float(loss_cfg.get("rmse_weight", 1.0)) * rmse
    )
    return {"total": total, "pcol": pcol, "scol": scol, "rmse": rmse}


def assert_finite_losses(losses: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor], targets: torch.Tensor) -> None:
    bad_losses = [name for name, value in losses.items() if not torch.isfinite(value).all()]
    bad_outputs = [name for name, value in outputs.items() if not torch.isfinite(value).all()]
    if not bad_losses and not bad_outputs:
        return

    target_min = int(targets.min().item())
    target_max = int(targets.max().item())
    details = {
        name: float(value.detach().float().cpu().nan_to_num().item())
        for name, value in losses.items()
        if value.ndim == 0
    }
    raise FloatingPointError(
        "Non-finite value detected during training. "
        f"bad_losses={bad_losses}, bad_outputs={bad_outputs}, "
        f"losses={details}, target_range=({target_min}, {target_max}). "
        "Try train.amp=false first; if it still happens, lower train.lr."
    )


def train_one_epoch(
    model,
    contrastive_loader,
    regression_loader,
    optimizer,
    scaler,
    device,
    amp_enabled,
    criteria,
    learnable_margins,
    optimization_parameters,
    cfg,
    epoch: int,
):
    model.train()
    meters = {name: AverageMeter() for name in ["total", "pcol", "scol", "rmse", "accuracy", "mae"]}
    progress = tqdm(contrastive_loader, desc=f"train {epoch}", leave=False)
    shared_batch = regression_loader is contrastive_loader
    regression_iter = None if shared_batch else iter(regression_loader)

    for contrastive_images, contrastive_targets, _ in progress:
        contrastive_sample_weights = torch.from_numpy(
            batch_inverse_frequency_sample_weights(
                contrastive_targets.numpy(),
                num_classes=int(cfg["model"].get("num_classes", 5)),
            )
        ).to(device, non_blocking=True)
        contrastive_images = contrastive_images.to(device, non_blocking=True)
        contrastive_targets = contrastive_targets.to(device, non_blocking=True).long()

        if shared_batch:
            regression_images = contrastive_images
            regression_targets = contrastive_targets
        else:
            try:
                regression_images, regression_targets, _ = next(regression_iter)
            except StopIteration:
                regression_iter = iter(regression_loader)
                regression_images, regression_targets, _ = next(regression_iter)
            regression_images = regression_images.to(device, non_blocking=True)
            regression_targets = regression_targets.to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp_enabled):
            contrastive_outputs = model(contrastive_images)
            regression_outputs = contrastive_outputs if shared_batch else model(regression_images)
        losses = compute_losses(
            contrastive_outputs,
            contrastive_targets,
            regression_outputs,
            regression_targets,
            contrastive_sample_weights,
            criteria,
            learnable_margins,
            cfg,
        )
        assert_finite_losses(losses, contrastive_outputs, contrastive_targets)
        if not shared_batch:
            assert_finite_losses(losses, regression_outputs, regression_targets)

        scaler.scale(losses["total"]).backward()
        max_grad_norm = cfg["train"].get("max_grad_norm")
        if max_grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(optimization_parameters, float(max_grad_norm))
        scaler.step(optimizer)
        scaler.update()

        batch_size = regression_images.size(0)
        metric_values = batch_metrics(
            regression_outputs["prediction"],
            regression_targets,
            int(cfg["model"].get("num_classes", 5)),
        )
        for name, loss in losses.items():
            meters[name].update(loss.item(), batch_size)
        meters["accuracy"].update(metric_values["accuracy"], batch_size)
        meters["mae"].update(metric_values["mae"], batch_size)
        progress.set_postfix(loss=f"{meters['total'].avg:.4f}", acc=f"{meters['accuracy'].avg:.3f}")

    return {name: meter.avg for name, meter in meters.items()}


@torch.no_grad()
def evaluate(model, loader, device, amp_enabled, cfg, description: str = "val"):
    model.eval()
    predictions: list[float] = []
    targets_all: list[int] = []

    for images, targets, _ in tqdm(loader, desc=description, leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).long()
        with autocast(enabled=amp_enabled):
            outputs = model(images)

        predictions.extend(outputs["prediction"].detach().cpu().float().tolist())
        targets_all.extend(targets.detach().cpu().long().tolist())

    metrics = aggregate_predictions(predictions, targets_all, int(cfg["model"].get("num_classes", 5)))
    return metrics


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    device = resolve_device(cfg.get("device", "auto"))
    output_dir = ensure_dir(cfg["train"]["output_dir"])
    contrastive_loader, regression_loader, val_loader, test_loader = make_dataloaders(cfg)

    model = HybridOrdinalNet(
        backbone=cfg["model"].get("backbone", "efficientnet_v2_s"),
        pretrained_imagenet=bool(cfg["model"].get("pretrained_imagenet", False)),
        projection_hidden_dim=int(cfg["model"].get("projection_hidden_dim", 1280)),
        projection_dim=int(cfg["model"].get("projection_dim", 128)),
        regression_hidden_dim=int(cfg["model"].get("regression_hidden_dim", 1280)),
        dropout=float(cfg["model"].get("dropout", 0.2)),
        regression_input=cfg["model"].get("regression_input", "backbone"),
    ).to(device)

    loss_cfg = cfg["loss"]
    num_classes = int(cfg["model"].get("num_classes", 5))
    criteria = {
        "pcol": PrototypeContrastiveOrdinalLoss(
            temperature=float(loss_cfg.get("temperature", 0.1)),
            num_classes=num_classes,
            margin_scale=float(loss_cfg.get("ordinal_margin_scale", 1.0)),
            normalize_ordinal_distance=bool(loss_cfg.get("normalize_ordinal_distance", False)),
            reduction=loss_cfg.get("reduction", "mean"),
        ),
        "scol": WeightedSupervisedContrastiveOrdinalLoss(
            temperature=float(loss_cfg.get("temperature", 0.1)),
            num_classes=num_classes,
            margin_scale=float(loss_cfg.get("ordinal_margin_scale", 1.0)),
            normalize_ordinal_distance=bool(loss_cfg.get("normalize_ordinal_distance", False)),
            reduction=loss_cfg.get("reduction", "mean"),
        ),
    }

    learnable_margins = build_learnable_margins(cfg, device)
    optimization_parameters = list(model.parameters())
    if learnable_margins is not None:
        optimization_parameters.extend(learnable_margins.parameters())
    optimizer = build_optimizer(optimization_parameters, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    amp_enabled = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)

    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    except ImportError:
        writer = None

    best_val_loss = float("inf")
    best_val_metrics = {}
    best_margin_state = {}
    best_epoch = 0
    bad_epochs = 0
    epochs = int(cfg["train"].get("epochs", 75))
    phase = "phase1_joint" if learnable_margins is not None else "fixed"
    margin_freeze_reason = None
    final_epoch = 0
    train_metrics = {}
    val_metrics = {}

    if learnable_margins is not None:
        initial_margins = margin_snapshot(learnable_margins, phase, margin_freeze_reason)
        print(
            "learnable adjacent margins initialized: "
            + ", ".join(f"{value:.4f}" for value in initial_margins["values"])
        )

    for epoch in range(1, epochs + 1):
        final_epoch = epoch
        train_metrics = train_one_epoch(
            model,
            contrastive_loader,
            regression_loader,
            optimizer,
            scaler,
            device,
            amp_enabled,
            criteria,
            learnable_margins,
            optimization_parameters,
            cfg,
            epoch,
        )
        val_metrics = evaluate(model, val_loader, device, amp_enabled, cfg)
        scheduler.step(float(val_metrics["rmse_loss"]))
        current_margin_state = margin_snapshot(
            learnable_margins,
            phase,
            margin_freeze_reason,
        )

        if writer is not None:
            for key, value in train_metrics.items():
                writer.add_scalar(f"train/{key}", value, epoch)
            for key, value in val_metrics.items():
                if isinstance(value, (float, int)):
                    writer.add_scalar(f"val/{key}", value, epoch)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)
            if learnable_margins is not None:
                for boundary, value in current_margin_state["by_boundary"].items():
                    writer.add_scalar(f"margins/{boundary}", value, epoch)
                writer.add_scalar(
                    "margins/phase",
                    1 if phase == "phase1_joint" else 2,
                    epoch,
                )

        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": cfg,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "ordinal_margins": (
                learnable_margins.state_dict() if learnable_margins is not None else None
            ),
            "margin_state": current_margin_state,
        }
        save_checkpoint(output_dir / "last.pt", payload)

        improved = float(val_metrics["rmse_loss"]) < best_val_loss
        if improved:
            best_val_loss = float(val_metrics["rmse_loss"])
            best_val_metrics = val_metrics
            best_margin_state = current_margin_state
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
                "best_margins": best_margin_state,
                "train": train_metrics,
                "val": val_metrics,
                "margins": current_margin_state,
            },
        )

        margin_text = ""
        if learnable_margins is not None:
            margin_text = " margins=[" + ", ".join(
                f"{value:.3f}" for value in current_margin_state["values"]
            ) + f"] phase={phase}"
        print(
            f"epoch {epoch:03d}: "
            f"train_loss={train_metrics['total']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_rmse={val_metrics['rmse_loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_mae={val_metrics['mae']:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
            f"{margin_text}"
        )

        if learnable_margins is not None and phase == "phase1_joint":
            loss_cfg = cfg["loss"]
            collapse_guard = float(loss_cfg.get("margin_collapse_guard", 0.1))
            accuracy_threshold = float(
                loss_cfg.get("margin_phase1_accuracy_threshold", 0.95)
            )
            phase1_epochs = int(loss_cfg.get("margin_phase1_epochs", 10))
            if current_margin_state["minimum"] <= collapse_guard:
                margin_freeze_reason = (
                    f"collapse_guard(min={current_margin_state['minimum']:.6f}, "
                    f"guard={collapse_guard:.6f})"
                )
            elif float(train_metrics["accuracy"]) >= accuracy_threshold:
                margin_freeze_reason = (
                    f"train_accuracy({train_metrics['accuracy']:.6f}>="
                    f"{accuracy_threshold:.6f})"
                )
            elif epoch >= phase1_epochs:
                margin_freeze_reason = f"max_phase1_epochs({phase1_epochs})"

            if margin_freeze_reason is not None:
                save_checkpoint(output_dir / "phase1_last.pt", payload)
                learnable_margins.freeze()
                phase = "phase2_frozen"
                bad_epochs = 0
                best_val_loss = float("inf")
                best_val_metrics = {}
                best_margin_state = {}
                best_epoch = 0
                initial_lr = float(cfg["train"].get("lr", 1e-3))
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = initial_lr
                scheduler = build_scheduler(optimizer, cfg)
                print(
                    "freezing learned margins and starting phase 2: "
                    f"{margin_freeze_reason}"
                )
                continue

        patience = int(cfg["train"].get("early_stopping_patience", 13))
        if phase != "phase1_joint" and bad_epochs >= patience:
            print(f"early stopping after {bad_epochs} epochs without validation improvement")
            break

    if writer is not None:
        writer.close()

    test_metrics = None
    if test_loader is not None:
        load_model_checkpoint(model, output_dir / "best.pt", strict=True)
        test_metrics = evaluate(model, test_loader, device, amp_enabled, cfg, description="test")
        print(
            f"outer test (best epoch {best_epoch:03d}): "
            f"rmse={test_metrics['rmse_loss']:.4f} "
            f"acc={test_metrics['accuracy']:.4f} "
            f"mae={test_metrics['mae']:.4f}"
        )

    save_json(
        output_dir / "metrics.json",
        {
            "epoch": final_epoch,
            "best_epoch": best_epoch,
            "best_val_rmse_loss": best_val_loss,
            "best_val": best_val_metrics,
            "best_margins": best_margin_state,
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
            "margins": margin_snapshot(
                learnable_margins,
                phase,
                margin_freeze_reason,
            ),
        },
    )


if __name__ == "__main__":
    main()
