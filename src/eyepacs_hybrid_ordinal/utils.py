from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * int(n)
        self.count += int(n)


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def worker_count(num_workers: int) -> int:
    if os.name == "nt":
        return min(int(num_workers), 2)
    return int(num_workers)


def build_optimizer(parameters, cfg: dict) -> torch.optim.Optimizer:
    name = cfg["train"].get("optimizer", "adam").lower()
    lr = float(cfg["train"]["lr"])
    weight_decay = float(cfg["train"].get("weight_decay", 0.0))
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer '{name}'.")


def class_count_string(targets: np.ndarray, num_classes: int) -> str:
    counts = np.bincount(targets.astype(int), minlength=num_classes)
    return ", ".join(f"{label}:{int(count)}" for label, count in enumerate(counts))
