"""Exercise public transfer learning for general, solvent, and ISA families.

Each source model has one target. The transferred model has two targets, so
the compatible backbone must load while the output head is reported as a shape
mismatch. Every transferred model is then reloaded through ``Analyzer``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from common import base_train_kwargs, workflow_workspace, write_tiny_dataset


CASES = (
    ("general", "GCN", False, False),
    ("solvent", "GCNwS", True, False),
    ("isa", "ISAT", False, True),
)


def _train_options(*, solvent: bool, isa: bool) -> dict:
    options = {}
    if solvent:
        options["molecule_columns"] = ["compound", "solvent"]
    if isa:
        options["sculptor_index"] = (6, 2, 0)
    return options


def _assert_transfer_report(path: Path, family: str) -> dict:
    report_path = path / "transfer_report.json"
    if not report_path.is_file():
        raise RuntimeError(f"{family} transfer did not write transfer_report.json.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["counts"]["loaded"] <= 0:
        raise RuntimeError(f"{family} transfer loaded no compatible parameters.")
    if report["counts"]["shape_mismatch"] <= 0:
        raise RuntimeError(
            f"{family} transfer did not report the intentionally resized output head."
        )
    return report


def run(output_root: str | None = None) -> dict:
    """Execute representative public transfer workflows for all manager families."""

    from D4CMPP2 import Analyzer, train

    results = {}
    with workflow_workspace("transfer-learning", output_root) as root:
        data_path = write_tiny_dataset(root)
        common = base_train_kwargs(root, data_path)
        for family, network_id, solvent, isa in CASES:
            options = _train_options(solvent=solvent, isa=isa)
            source_path = root / f"source-{family}"
            train(
                **common,
                **options,
                network=network_id,
                target=["target_a"],
                MODEL_PATH=str(source_path),
            )
            target_path = root / f"transferred-{family}"
            trained_path = train(
                **common,
                **options,
                TRANSFER_PATH=str(source_path),
                network=network_id,
                target=["target_a", "target_b"],
                MODEL_PATH=str(target_path),
            )
            report = _assert_transfer_report(target_path, family)
            analyzer = Analyzer(trained_path, save_result=False, device="cpu")
            inputs = {"compound": ["CC", "CCC"]}
            if solvent:
                inputs["solvent"] = ["O", "CO"]
            rows = analyzer.predict_rows(**inputs)
            if [row.status for row in rows] != ["ok", "ok"]:
                raise RuntimeError(f"{family} transferred Analyzer returned invalid rows.")
            predictions = [row.prediction.tolist() for row in rows]
            if any(
                len(values) != 2 or not all(math.isfinite(float(value)) for value in values)
                for values in predictions
            ):
                raise RuntimeError(
                    f"{family} transferred Analyzer returned invalid predictions {predictions}."
                )
            results[family] = {
                "network": network_id,
                "source_path": str(source_path),
                "model_path": str(target_path),
                "loaded_parameters": report["counts"]["loaded"],
                "shape_mismatches": report["counts"]["shape_mismatch"],
                "prediction_shape": [len(rows), len(predictions[0])],
            }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root")
    args = parser.parse_args()
    print(json.dumps(run(args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
