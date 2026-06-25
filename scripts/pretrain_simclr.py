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

from eyepacs_contrastive.data import EyePACSDataset
from eyepacs_contrastive.losses import NTXentLoss
from eyepacs_contrastive.models import SimCLRResNet
from eyepacs_contrastive.transforms import make_simclr_transform
from eyepacs_contrastive.utils import AverageMeter, ensure_dir, load_config, resolve_device, save_checkpoint, set_seed, worker_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain a ResNet encoder with SimCLR on EyePACS.")
    parser.add_argument("--config", default="configs/simclr_resnet18_eyepacs.yaml", help="Path to YAML config.")
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


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    device = resolve_device(cfg.get("device", "auto"))
    output_dir = ensure_dir(cfg["train"]["output_dir"])

    transform = make_simclr_transform(image_size=int(cfg["data"]["image_size"]))
    dataset = EyePACSDataset(
        labels_csv=cfg["data"]["labels_csv"],
        image_root=cfg["data"]["image_root"],
        image_col=cfg["data"].get("image_col"),
        label_col=cfg["data"].get("label_col"),
        max_samples=cfg["data"].get("max_samples"),
        transform=transform,
        contrastive=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["data"]["batch_size"]),
        shuffle=True,
        num_workers=worker_count(int(cfg["data"].get("num_workers", 0))),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    if len(loader) == 0:
        raise ValueError(
            "No pretraining batches were created. Use more samples or a smaller batch_size; "
            "SimCLR also requires batch_size > 1."
        )

    model = SimCLRResNet(
        backbone=cfg["model"].get("backbone", "resnet18"),
        pretrained_imagenet=bool(cfg["model"].get("pretrained_imagenet", False)),
        projection_dim=int(cfg["model"].get("projection_dim", 128)),
        hidden_dim=int(cfg["model"].get("hidden_dim", 2048)),
    ).to(device)
    criterion = NTXentLoss(temperature=float(cfg["train"].get("temperature", 0.5)))
    optimizer = build_optimizer(model.parameters(), cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(cfg["train"]["epochs"]) * len(loader)),
    )

    amp_enabled = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)

    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    except ImportError:
        writer = None

    best_loss = float("inf")
    global_step = 0
    epochs = int(cfg["train"]["epochs"])

    for epoch in range(1, epochs + 1):
        model.train()
        losses = AverageMeter()
        progress = tqdm(loader, desc=f"epoch {epoch}/{epochs}", leave=False)

        for view1, view2, _, _ in progress:
            view1 = view1.to(device, non_blocking=True)
            view2 = view2.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp_enabled):
                _, z1 = model(view1)
                _, z2 = model(view2)
                loss = criterion(z1, z2)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            losses.update(loss.item(), view1.size(0))
            progress.set_postfix(loss=f"{losses.avg:.4f}")

            if writer is not None:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
            global_step += 1

        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": cfg,
            "loss": losses.avg,
        }
        save_checkpoint(output_dir / "last.pt", payload)
        if losses.avg < best_loss:
            best_loss = losses.avg
            save_checkpoint(output_dir / "best.pt", payload)

        print(f"epoch {epoch:03d}: loss={losses.avg:.4f} best={best_loss:.4f}")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
