"""Train the teammate's Balanced Softmax Hybrid model under the held-out IDRiD protocol."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import train_idrid_fixed_iterations as heldout
from eyepacs_hybrid_ordinal.utils import load_config

DEFAULT_CONFIG = "configs/hybrid_idrid_balanced_softmax.yaml"


def run(cfg, *, allow_synthetic=False):
    if not cfg["model"].get("balanced_softmax", False):
        raise ValueError("This runner requires model.balanced_softmax: true.")
    weight = float(cfg["loss"].get("classification_weight", -1))
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("classification_weight must be finite and nonnegative.")
    return heldout.run(cfg, allow_synthetic=allow_synthetic)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--classification-weight", type=float, help="Override the branch YAML's default 0.1 BS weight")
    args = parser.parse_args()
    cfg = load_config(args.config)
    for name, section, key in (
        ("data_dir", "data", "data_dir"), ("output_dir", "train", "output_dir"),
        ("num_workers", "data", "num_workers"), ("epochs", "train", "epochs"),
    ):
        value = getattr(args, name)
        if value is not None:
            cfg[section][key] = str(value.resolve()) if isinstance(value, Path) else value
    if args.device is not None:
        cfg["device"] = args.device
    if args.classification_weight is not None:
        cfg["loss"]["classification_weight"] = args.classification_weight
        if args.output_dir is None:
            tag = str(args.classification_weight).replace(".", "p")
            cfg["train"]["output_dir"] = f"checkpoints/hybrid_idrid_bs_w{tag}_train_only_fixed22_gem_no_spatial_seed{cfg['seed']}"
    run(cfg)


if __name__ == "__main__":
    main()
