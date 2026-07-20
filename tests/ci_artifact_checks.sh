#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dist_dir="$(mktemp -d)"
trap 'rm -rf "$dist_dir"' EXIT

cd "$repo_root"
python -m pip install --quiet build==1.3.0
python -m build --outdir "$dist_dir"
python tests/check_distribution_artifacts.py "$dist_dir"
python tests/wheel_install_smoke.py
