import json
import math
import os
from pathlib import Path
import shutil
import sys
import unittest

from fixtures import ROOT, TINY_REGRESSION_CSV, isolated_workdir
from markers import heavy_test


BASELINE_PATH = ROOT / "tests" / "baselines" / "gcn_cpu.json"
EXPECTED_MODEL_FILES = {
    "config.yaml",
    "final.pth",
    "model_summary.txt",
    "network.py",
    "network_refer.yaml",
    "scaler.pkl",
}
EXPECTED_RESULT_FILES = {
    "learning_curve.csv",
    "learning_curve.png",
    "metrics.csv",
    "prediction.csv",
    "prediction.png",
}


class GcnEndToEndTests(unittest.TestCase):
    @heavy_test
    def test_cpu_training_reload_and_prediction(self):
        """Exercise the current backend; run after workspace .venv setup."""
        required = ("torch", "rdkit", "pandas", "numpy", "sklearn", "yaml")
        missing = []
        import importlib.util

        for dependency in required:
            if importlib.util.find_spec(dependency) is None:
                missing.append(dependency)
        graph_backends = [name for name in ("torch_geometric",) if importlib.util.find_spec(name)]
        if missing:
            self.fail(
                "D4CMPP2_RUN_HEAVY=1 but the .venv is incomplete; "
                f"missing={missing}, graph_backends={graph_backends}"
            )

        if not graph_backends:
            self.skipTest("no graph backend is installed; install PyG for the planned migration")

        if "torch_geometric" not in graph_backends:
            self.fail("PyG is required by the primary backend but torch_geometric is not installed")

        package_parent = str(ROOT.parent)
        if package_parent not in sys.path:
            sys.path.insert(0, package_parent)

        from D4CMPP2 import Analyzer, train
        import torch

        torch.manual_seed(42)

        with isolated_workdir() as temporary:
            data_path = temporary / "tiny_regression.csv"
            shutil.copyfile(TINY_REGRESSION_CSV, data_path)
            model_dir = temporary / "_Models"
            graph_dir = temporary / "_Graphs"
            model_dir.mkdir()
            graph_dir.mkdir()

            model_path = train(
                network="GCN",
                data=str(data_path),
                target=["target_a"],
                device="cpu",
                max_epoch=2,
                batch_size=4,
                hidden_dim=8,
                conv_layers=1,
                linear_layers=1,
                dropout=0.0,
                lr_patience=2,
                early_stopping_patience=2,
                split_random_seed=42,
                num_workers=0,
                MODEL_DIR=str(model_dir),
                GRAPH_DIR=str(graph_dir),
                NET_DIR=str(ROOT / "networks"),
                NET_REFER=str(ROOT / "network_refer.yaml"),
            )

            self.assertIsNotNone(model_path, "train() swallowed an exception or returned no model path")
            model_path = Path(model_path)
            self.assertTrue(model_path.is_dir())
            self.assertTrue(EXPECTED_MODEL_FILES.issubset({entry.name for entry in model_path.iterdir()}))
            result_path = model_path / "result"
            self.assertTrue(result_path.is_dir())
            self.assertTrue(EXPECTED_RESULT_FILES.issubset({entry.name for entry in result_path.iterdir()}))

            analyzer = Analyzer.MolAnalyzer_v2(model_path.name, save_result=False)
            single = analyzer.predict(compound="CCO")
            multiple = analyzer.predict(compound=["CC", "CCC"])

            self.assertEqual(list(single), [("CCO",)])
            self.assertEqual(list(multiple), [("CC",), ("CCC",)])
            observed = {
                "CCO": float(single[("CCO",)][0]),
                "CC": float(multiple[("CC",)][0]),
                "CCC": float(multiple[("CCC",)][0]),
            }
            self.assertTrue(all(math.isfinite(value) for value in observed.values()))

            duplicate_and_invalid = ["CC", "CC", "not-a-smiles"]
            structured = analyzer.predict_rows(compound=duplicate_and_invalid)
            self.assertEqual(duplicate_and_invalid, ["CC", "CC", "not-a-smiles"])
            self.assertEqual([row.row_index for row in structured], [0, 1, 2])
            self.assertEqual([row.status for row in structured], ["ok", "ok", "invalid"])
            self.assertAlmostEqual(
                float(structured[0].prediction[0]),
                float(structured[1].prediction[0]),
                delta=1e-7,
            )
            self.assertIn("compound", structured[2].error)

            import pandas as pd

            inference_csv = temporary / "inference.csv"
            pd.DataFrame(
                {"sample_id": ["a", "b", "c"], "compound": duplicate_and_invalid}
            ).to_csv(inference_csv, index=False)
            output_csv = analyzer.predict_csv(
                inference_csv,
                temporary / "inference_prediction.csv",
                index_col="sample_id",
                uncertainty_samples=3,
                uncertainty_seed=17,
            )
            batch_result = pd.read_csv(output_csv)
            self.assertEqual(batch_result["row_index"].tolist(), ["a", "b", "c"])
            self.assertEqual(
                batch_result["prediction_status"].tolist(),
                ["ok", "ok", "invalid"],
            )
            self.assertIn("target_a_pred_std", batch_result.columns)
            self.assertEqual(batch_result["uncertainty_samples"].tolist(), [3, 3, 3])

            uncertainty_a = analyzer.predict_uncertainty(
                compound=["CC", "CCC"], samples=3, seed=17
            )
            uncertainty_b = analyzer.predict_uncertainty(
                compound=["CC", "CCC"], samples=3, seed=17
            )
            for row_a, row_b in zip(uncertainty_a.mean, uncertainty_b.mean):
                self.assertAlmostEqual(
                    float(row_a.prediction[0]),
                    float(row_b.prediction[0]),
                    delta=1e-7,
                )
            self.assertFalse(analyzer.nm.network.training)
            if os.environ.get("D4CMPP2_SHOW_BASELINE") == "1":
                print(f"GCN CPU observed predictions: {json.dumps(observed, sort_keys=True)}")

            baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
            if baseline["status"] == "recorded":
                tolerance = float(baseline["tolerance"])
                for smiles, expected in baseline["predictions"].items():
                    self.assertAlmostEqual(observed[smiles], float(expected), delta=tolerance)

            self.assertTrue(model_path.is_relative_to(temporary))
            self.assertTrue(graph_dir.is_relative_to(temporary))

            latest = model_path / "checkpoints" / "latest.ckpt"
            best = model_path / "checkpoints" / "best.ckpt"
            self.assertTrue(latest.is_file())
            self.assertTrue(best.is_file())
            manifests = list((model_path / "runs").glob("*/run_manifest.json"))
            self.assertEqual(len(manifests), 1)
            self.assertEqual(
                json.loads(manifests[0].read_text(encoding="utf-8"))["status"],
                "completed",
            )

            resumed_path = train(
                RESUME_PATH=str(model_path),
                device="cpu",
                max_epoch=1,
                num_workers=0,
            )
            self.assertEqual(Path(resumed_path).resolve(), model_path.resolve())
            import pandas as pd

            curve = pd.read_csv(model_path / "result" / "learning_curve.csv")
            self.assertEqual(len(curve), 3)
            checkpoint = torch.load(
                model_path / "checkpoints" / "latest.ckpt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(checkpoint["completed_epoch"], 2)
            self.assertEqual(checkpoint["next_epoch"], 3)
            manifests = list((model_path / "runs").glob("*/run_manifest.json"))
            self.assertEqual(len(manifests), 2)
            self.assertEqual(
                sorted(
                    json.loads(path.read_text(encoding="utf-8"))["mode"]
                    for path in manifests
                ),
                ["resume", "scratch"],
            )


if __name__ == "__main__":
    unittest.main()
