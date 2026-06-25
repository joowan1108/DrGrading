from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eyepacs_contrastive.data import EyePACSDataset, make_train_val_indices
from eyepacs_contrastive.models import LinearEvalModel, load_simclr_encoder
from eyepacs_contrastive.transforms import make_eval_transform, make_supervised_train_transform
from eyepacs_contrastive.utils import (
    AverageMeter,
    accuracy,
    class_weights_from_targets,
    ensure_dir,
    load_config,
    resolve_device,
    save_checkpoint,
    save_json,
    set_seed,
    worker_count,
)

try:
    from sklearn.metrics import cohen_kappa_score
except ImportError:
    cohen_kappa_score = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Linear evaluation for a SimCLR-pretrained EyePACS encoder.")
    parser.add_argument("--config", default="configs/linear_eval_resnet18.yaml", help="Path to YAML config.")
    return parser.parse_args()


def build_optimizer(parameters, cfg: dict) -> torch.optim.Optimizer:
    name = cfg["train"].get("optimizer", "adamw").lower()
    lr = float(cfg["train"]["lr"])
    weight_decay = float(cfg["train"].get("weight_decay", 0.0))
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer '{name}'.")


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, amp_enabled: bool) -> dict[str, float]:
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    criterion = torch.nn.CrossEntropyLoss()
    all_targets: list[int] = []
    all_predictions: list[int] = []

    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast(enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)

        batch_size = images.size(0)
        losses.update(loss.item(), batch_size)
        accuracies.update(accuracy(logits, targets), batch_size)
        all_targets.extend(targets.cpu().numpy().tolist())
        all_predictions.extend(logits.argmax(dim=1).cpu().numpy().tolist())

    metrics = {"loss": losses.avg, "accuracy": accuracies.avg}
    if cohen_kappa_score is not None and len(set(all_targets)) > 1:
        metrics["quadratic_kappa"] = float(cohen_kappa_score(all_targets, all_predictions, weights="quadratic"))
    else:
        metrics["quadratic_kappa"] = float("nan")
    return metrics


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    device = resolve_device(cfg.get("device", "auto"))
    output_dir = ensure_dir(cfg["train"]["output_dir"])
    image_size = int(cfg["data"]["image_size"])
    num_classes = int(cfg["model"].get("num_classes", 5))

    train_dataset = EyePACSDataset(
        labels_csv=cfg["data"]["labels_csv"],
        image_root=cfg["data"]["image_root"],
        image_col=cfg["data"].get("image_col"),
        label_col=cfg["data"].get("label_col"),
        max_samples=cfg["data"].get("max_samples"),
        transform=make_supervised_train_transform(image_size=image_size),
    )
    val_dataset = EyePACSDataset(
        labels_csv=cfg["data"]["labels_csv"],
        image_root=cfg["data"]["image_root"],
        image_col=cfg["data"].get("image_col"),
        label_col=cfg["data"].get("label_col"),
        max_samples=cfg["data"].get("max_samples"),
        transform=make_eval_transform(image_size=image_size),
    )

    train_idx, val_idx = make_train_val_indices(train_dataset.targets, float(cfg["data"].get("val_size", 0.2)), int(cfg.get("seed", 42)))
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)

    loader_kwargs = {
        "batch_size": int(cfg["data"]["batch_size"]),
        "num_workers": worker_count(int(cfg["data"].get("num_workers", 0))),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_subset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_subset, shuffle=False, **loader_kwargs)

    model = LinearEvalModel(
        backbone=cfg["model"].get("backbone", "resnet18"),
        num_classes=num_classes,
        pretrained_imagenet=False,
        freeze_backbone=bool(cfg["model"].get("freeze_backbone", True)),
    ).to(device)
    load_simclr_encoder(model, cfg["model"]["checkpoint"])

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = build_optimizer(trainable_parameters, cfg)
    amp_enabled = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)

    if bool(cfg["train"].get("class_weighted_loss", True)):
        class_weights = class_weights_from_targets(train_dataset.targets[train_idx], num_classes).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    best_score = -1.0
    best_kappa = -1.0
    best_accuracy = 0.0
    epochs = int(cfg["train"]["epochs"])

    for epoch in range(1, epochs + 1):
        model.train()
        losses = AverageMeter()
        accuracies = AverageMeter()
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False)

        for images, targets, _ in progress:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = images.size(0)
            losses.update(loss.item(), batch_size)
            accuracies.update(accuracy(logits, targets), batch_size)
            progress.set_postfix(loss=f"{losses.avg:.4f}", acc=f"{accuracies.avg:.4f}")

        val_metrics = evaluate(model, val_loader, device, amp_enabled)
        score = val_metrics["quadratic_kappa"]
        if np.isnan(score):
            score = val_metrics["accuracy"]

        is_best = score > best_score
        best_score = max(best_score, score)
        if not np.isnan(val_metrics["quadratic_kappa"]):
            best_kappa = max(best_kappa, val_metrics["quadratic_kappa"])
        best_accuracy = max(best_accuracy, val_metrics["accuracy"])

        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "val_metrics": val_metrics,
        }
        save_checkpoint(output_dir / "last.pt", payload)
        if is_best:
            save_checkpoint(output_dir / "best.pt", payload)

        metrics_path = output_dir / "metrics.json"
        save_json(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": losses.avg,
                "train_accuracy": accuracies.avg,
                "val": val_metrics,
                "best_score": best_score,
                "best_quadratic_kappa": best_kappa,
                "best_accuracy": best_accuracy,
            },
        )
        print(
            "epoch "
            f"{epoch:03d}: train_loss={losses.avg:.4f} train_acc={accuracies.avg:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_qwk={val_metrics['quadratic_kappa']:.4f}"
        )


if __name__ == "__main__":
    main()
