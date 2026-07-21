"""Train, save, reload, and predict with every solvent-aware model.

Run directly with temporary outputs::

    python examples/integration/solvent_analyzer.py

Keep artifacts for inspection::

    python examples/integration/solvent_analyzer.py --output-root integration-output/solvent
"""

from __future__ import annotations

import argparse
import json
import math

from common import base_train_kwargs, workflow_workspace, write_tiny_dataset


NETWORK_IDS = ("GCNwS", "MPNNwS", "DMPNNwS", "AFPwS", "GATwS")


def run(output_root: str | None = None) -> dict:
    """Execute the five-model public train-to-Analyzer workflow."""

    from D4CMPP2 import Analyzer, train
    from rdkit import rdBase

    results = {}
    with workflow_workspace("solvent-analyzer", output_root) as root:
        data_path = write_tiny_dataset(root)
        common = base_train_kwargs(root, data_path)
        for network_id in NETWORK_IDS:
            model_path = root / f"saved-{network_id}"
            trained_path = train(
                **common,
                network=network_id,
                target=["target_a"],
                molecule_columns=["compound", "solvent"],
                MODEL_PATH=str(model_path),
            )
            required = ("network.py", "config.yaml", "final.pth", "scaler.pkl")
            missing = [name for name in required if not (model_path / name).is_file()]
            if missing:
                raise RuntimeError(
                    f"{network_id} completed training but is missing artifacts {missing}."
                )
            analyzer = Analyzer(trained_path, save_result=False, device="cpu")
            # Invalid input is intentional here. Keep the Analyzer's structured
            # error result while suppressing RDKit's expected parser diagnostics.
            with rdBase.BlockLogs():
                rows = analyzer.predict_rows(
                    compound=["CC", "CC", "not-a-smiles"],
                    solvent=["O", "O", "O"],
                )
            statuses = [row.status for row in rows]
            if statuses != ["ok", "ok", "invalid"]:
                raise RuntimeError(
                    f"{network_id} returned unexpected row statuses {statuses}."
                )
            first = float(rows[0].prediction[0])
            duplicate = float(rows[1].prediction[0])
            if not math.isfinite(first) or abs(first - duplicate) > 1e-6:
                raise RuntimeError(
                    f"{network_id} returned inconsistent duplicate predictions "
                    f"{first} and {duplicate}."
                )
            results[network_id] = {
                "model_path": str(model_path),
                "prediction": first,
                "invalid_error": rows[2].error,
            }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root")
    args = parser.parse_args()
    print(json.dumps(run(args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
