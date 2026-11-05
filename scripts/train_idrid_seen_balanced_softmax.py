"""Clone the 82.52% IDRiD train+test protocol, add BS, then re-infer and plot."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import train_idrid_all_data as all_data
import idrid_seen_class1_tsne as analysis
from eyepacs_hybrid_ordinal.utils import save_json

ROOT = all_data.hybrid.ROOT


def source_config(source_run):
    source_run = Path(source_run).expanduser().resolve()
    recorded = json.loads((source_run / "run_config.json").read_text())
    cfg = copy.deepcopy(recorded["config"])
    if recorded.get("official_test_used_for_training") is not True or cfg["model"].get("balanced_softmax", False):
        raise ValueError("--source-run must be the original non-BS train+test experiment.")
    if int(cfg["data"]["image_size"]) != 224 or cfg["model"].get("pooling") != "gem" or cfg["model"].get("spatial_attention", False):
        raise ValueError("Expected the original 224-input / GeM / no-spatial IDRiD source.")
    values, origin = None, None
    initial = source_run / "initial_margins.json"
    if initial.is_file():
        values = json.loads(initial.read_text()).get("values")
        origin = str(initial)
    if values is None:
        metrics = source_run / "evaluation_metrics.json"
        if metrics.is_file():
            values = json.loads(metrics.read_text()).get("margins", {}).get("initial_values")
            origin = str(metrics)
    if values is None:
        values = cfg["loss"].get("margin_init_values")
        origin = str(source_run / "run_config.json")
    if values is None:
        raise ValueError("Source initial margins are missing. Use the original run containing initial_margins.json.")
    if len(values) != int(cfg["model"]["num_classes"])-1 or not all(math.isfinite(v) and v > cfg["loss"].get("margin_min", 0) for v in values):
        raise ValueError("Invalid recorded initial margins.")
    return cfg, values, origin


def configure(source_run, data_dir, output_dir, *, classification_weight=0.1, device="auto", num_workers=None,
              epochs=None, test_every_epochs=None, test_every_steps=None):
    if not math.isfinite(classification_weight) or classification_weight < 0:
        raise ValueError("classification-weight must be finite and nonnegative.")
    cfg, initial_values, origin = source_config(source_run)
    source = copy.deepcopy(cfg)
    if test_every_epochs is not None and test_every_steps is not None:
        raise ValueError("Choose --test-every-epochs or --test-every-steps.")
    for value in (epochs, test_every_epochs, test_every_steps):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ValueError("Epoch budget and evaluation interval must be positive integers.")
    if epochs is not None:
        cfg["train"]["epochs"] = epochs
    if int(cfg["train"]["epochs"]) <= int(cfg["loss"]["margin_phase1_max_epochs"]):
        raise ValueError("Total epochs must exceed margin_phase1_max_epochs so Phase 2 can run.")
    for key, value in [("test_every_epochs", test_every_epochs), ("test_every_steps", test_every_steps)]:
        if value is not None:
            cfg["train"][key] = value
    cfg["device"] = device
    cfg["data"]["data_dir"] = str(Path(data_dir).resolve())
    if num_workers is not None:
        cfg["data"]["num_workers"] = num_workers
    cfg["model"]["balanced_softmax"] = True
    cfg["loss"]["classification_weight"] = classification_weight
    cfg["loss"].setdefault("mmnp_pair_reduction", "mean")
    # BS adds a randomly initialized head. Reuse saved initial gaps to keep them matched.
    cfg["loss"].update(margin_init_values=initial_values, margin_init_jitter=0)
    cfg["loss"].pop("margin_init_min", None)
    cfg["loss"].pop("margin_init_max", None)
    cfg["train"]["output_dir"] = str(Path(output_dir).resolve())
    cfg["provenance"] = {"source_run": str(Path(source_run).resolve()), "source_weights_loaded": False,
                         "source_initial_margins": origin, "added_balanced_softmax_weight": classification_weight,
                         "matched_settings": "source architecture, data augmentation, seed, alpha/beta, RMSE/MMNP, optimizer, scheduler, phase boundaries and initial gaps",
                         "epoch_budget": {"source": int(source["train"]["epochs"]), "effective": int(cfg["train"]["epochs"]),
                                          "changed": int(source["train"]["epochs"]) != int(cfg["train"]["epochs"])},
                         "prediction_source": "round_and_clip_regression; BS head is auxiliary"}
    return cfg, source


def run(source_run, data_dir, output_dir, *, classification_weight=0.1, device="auto", num_workers=None,
        tsne_options=None, sweep=False, allow_synthetic=False, epochs=None,
        test_every_epochs=None, test_every_steps=None):
    cfg, original = configure(source_run, data_dir, output_dir, classification_weight=classification_weight,
                              device=device, num_workers=num_workers, epochs=epochs,
                              test_every_epochs=test_every_epochs, test_every_steps=test_every_steps)
    print(f"BS comparison: input={cfg['data']['image_size']} epochs={cfg['train']['epochs']} "
          f"alpha/beta={cfg['loss']['alpha']}/{cfg['loss']['beta']} margin_lr={cfg['loss']['margin_lr']} "
          f"MMNP={cfg['loss']['mmnp_pair_reduction']} | official train+test", flush=True)
    result = all_data.run(cfg, allow_synthetic=allow_synthetic, allow_test_training=True)
    output = Path(cfg["train"]["output_dir"])
    save_json(output / "source_configuration_comparison.json", {"original_config": original, "bs_config": cfg})
    analysis.analyze(output, Path(data_dir), output / "analysis_class1_all", device=device,
                     tsne_options=tsne_options, sweep=sweep, allow_synthetic=allow_synthetic)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=ROOT / "checkpoints/hybrid_idrid_all_data_gem_no_spatial_seed42")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/idrid")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints/hybrid_idrid_all_data_gem_bs_matched_seed42")
    parser.add_argument("--classification-weight", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--epochs", type=int, help="Override the source training budget; must exceed the Phase 1 maximum.")
    period = parser.add_mutually_exclusive_group()
    period.add_argument("--test-every-epochs", type=int, help="Report training-seen test metrics every N epochs.")
    period.add_argument("--test-every-steps", type=int, help="Report training-seen test metrics every N mini-batch iterations.")
    parser.add_argument("--scaling", choices=["none", "standard", "whiten"], default="none")
    parser.add_argument("--perplexity", type=float, default=30)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tsne-sweep", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    return run(args.source_run, args.data_dir, args.output_dir, classification_weight=args.classification_weight,
               device=args.device, num_workers=args.num_workers,
               epochs=args.epochs, test_every_epochs=args.test_every_epochs, test_every_steps=args.test_every_steps,
               tsne_options={"scaling": args.scaling, "perplexity": args.perplexity, "max_iter": args.max_iter},
               sweep=args.tsne_sweep)


if __name__ == "__main__":
    main()
