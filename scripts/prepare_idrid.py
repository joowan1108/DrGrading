"""Run this script on the remote training computer to download/prepare IDRiD."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eyepacs_hybrid_ordinal.idrid import download_idrid, prepare_idrid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/idrid"))
    parser.add_argument("--source-dir", type=Path, help="Existing extracted Kaggle dataset root")
    parser.add_argument("--download", action="store_true", help="Download HERE (run on the remote host)")
    args = parser.parse_args()
    if args.download and args.source_dir:
        parser.error("Use either --download or --source-dir.")
    source = download_idrid(args.data_dir) if args.download else args.source_dir or args.data_dir / "raw"
    manifest = prepare_idrid(source, args.data_dir / "prepared")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
