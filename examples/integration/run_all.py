"""Run every heavy integration example in an isolated subprocess."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
WORKFLOWS = (
    ("solvent-analyzer", HERE / "solvent_analyzer.py"),
    ("transfer-learning", HERE / "transfer_learning.py"),
)


def run_all(base: Path, selected: set[str]) -> int:
    """Run selected workflows and return a process-compatible status code."""

    summary = []
    for name, script in WORKFLOWS:
        if selected and name not in selected:
            continue
        output_root = base / name
        working_directory = base / f".{name}-cwd"
        output_root.mkdir(parents=True, exist_ok=True)
        working_directory.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(script),
            "--output-root",
            str(output_root),
        ]
        print(f"\n=== Running {name} ===", flush=True)
        started = time.monotonic()
        completed = subprocess.run(command, cwd=working_directory, check=False)
        elapsed = time.monotonic() - started
        summary.append((name, completed.returncode, elapsed, output_root))

    print("\n=== Integration summary ===")
    for name, returncode, elapsed, output_root in summary:
        status = "PASS" if returncode == 0 else "FAIL"
        print(f"{status:4} {name:20} {elapsed:7.2f}s  {output_root}")
    failures = [name for name, returncode, _, _ in summary if returncode != 0]
    if failures:
        print(f"Failed workflows: {failures}", file=sys.stderr)
        return 1
    return 0


def main() -> None:
    names = {name for name, _ in WORKFLOWS}
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        action="append",
        choices=sorted(names),
        default=[],
        help="Run only this workflow; repeat the option to select more than one.",
    )
    parser.add_argument(
        "--keep-output",
        type=Path,
        help="Keep generated models and reports under this directory.",
    )
    args = parser.parse_args()

    if args.keep_output is not None:
        base = args.keep_output.resolve()
        base.mkdir(parents=True, exist_ok=True)
        raise SystemExit(run_all(base, set(args.only)))

    with tempfile.TemporaryDirectory(prefix="d4cmpp2-integration-") as temporary:
        base = Path(temporary).resolve()
        status = run_all(base, set(args.only))
        print(f"Temporary outputs removed from {base}")
        raise SystemExit(status)


if __name__ == "__main__":
    main()
