"""Export the specified spatial map, then run both IDRiD training/evaluation protocols."""
from __future__ import annotations

import argparse
from pathlib import Path

import export_spatial_attention as spatial
import compare_idrid_best_eyepacs as comparison


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    spatial_source = parser.add_mutually_exclusive_group()
    spatial_source.add_argument("--spatial-run", default=f"checkpoints/{spatial.DEFAULT_SPATIAL_RUN}")
    spatial_source.add_argument("--spatial-checkpoint")
    parser.add_argument("--eyepacs-labels-csv")
    parser.add_argument("--eyepacs-image-root")
    parser.add_argument("--spatial-sample-index", type=int, default=0)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--source-config", type=Path, default=comparison.SOURCE)
    source.add_argument("--source-run", type=Path)
    parser.add_argument("--data-dir", type=Path, default=comparison.ROOT / "data/idrid")
    parser.add_argument("--output-dir", type=Path, default=comparison.ROOT / "outputs/spatial_and_idrid_best_seed42")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batches-per-epoch", type=int, default=22)
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--skip-spatial", action="store_true", help="Run only the IDRiD comparison.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    source, provenance = comparison.read_source(args.source_config, args.source_run)
    prepared = args.data_dir.resolve() / "prepared/manifest.json"
    if not prepared.is_file():
        raise FileNotFoundError(f"Existing IDRiD prepared data not found: {prepared}. Set --data-dir to the previously prepared IDRiD folder.")
    if not args.skip_spatial:
        options = (["--checkpoint", args.spatial_checkpoint] if args.spatial_checkpoint
                   else ["--run-dir", args.spatial_run])
        options += ["--class-id", "3", "--split", "val", "--sample-index", str(args.spatial_sample_index),
                    "--device", args.device, "--output-dir", str(args.output_dir / "spatial_class3")]
        for flag, value in [("--labels-csv", args.eyepacs_labels_csv), ("--image-root", args.eyepacs_image_root)]:
            if value is not None:
                options += [flag, value]
        print("[1/2] Spatial Attention: one ground-truth Class 3 EyePACS validation image", flush=True)
        spatial.run(spatial.parse_args(options))
    print("[2/2] IDRiD: train+test / train-only comparison with the selected non-BS configuration", flush=True)
    return comparison.run_comparison(
        source, provenance, data_dir=args.data_dir, output_dir=args.output_dir / "idrid_comparison",
        device=args.device, epochs=args.epochs, batches_per_epoch=args.batches_per_epoch,
        num_workers=args.num_workers, seed=args.seed, skip_completed=args.skip_completed,
    )


if __name__ == "__main__":
    main()
