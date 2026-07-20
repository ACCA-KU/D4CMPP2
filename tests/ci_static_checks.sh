#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python -m pip install --quiet ruff==0.15.3 pyright==1.1.410
typed_modules=(
  src/DataManager/contracts.py
  src/TrainManager/callbacks.py
  src/utils/config_resolution.py
  src/utils/output.py
)

python -m ruff check --select F63,F7,F82 "${typed_modules[@]}"
pyright --project pyrightconfig.json
