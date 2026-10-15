#!/usr/bin/env bash
# Invoke on the remote GPU machine. No SSH or dataset transfer to your local PC.
set -euo pipefail
task_repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_repo_dir"
task_python="${IDRID_PYTHON:-python3}"
task_data_dir="${IDRID_DATA_DIR:-data/idrid}"
"$task_python" scripts/prepare_idrid.py --data-dir "$task_data_dir" --download
"$task_python" scripts/train_idrid.py --data-dir "$task_data_dir" "$@"
