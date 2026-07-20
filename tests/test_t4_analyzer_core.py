import ast
import tempfile
import unittest
from pathlib import Path

from fixtures import ROOT, load_source_module
from markers import heavy_test


class AnalyzerStaticContractTests(unittest.TestCase):
    def test_positive_only_isa_source_has_no_pn_branches(self):
        source = (ROOT / "src/Analyzer/ISAAnalyzer.py").read_text(encoding="utf-8")
        self.assertNotIn("negative", source)
        self.assertNotIn("feature_N", source)
        pn_source = (ROOT / "src/Analyzer/ISAPNAnalyzer.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"negative"', pn_source)
        self.assertIn('"feature_N"', pn_source)

    def test_isapn_analyzer_inherits_isa_analyzer(self):
        tree = ast.parse(
            (ROOT / "src/Analyzer/ISAPNAnalyzer.py").read_text(encoding="utf-8")
        )
        classes = {
            node.name: {
                base.id for base in node.bases if isinstance(base, ast.Name)
            }
            for node in tree.body
            if isinstance(node, ast.ClassDef)
        }
        self.assertIn("ISAAnalyzer", classes["ISAPNAnalyzer"])
        self.assertIn("ISAAnalyzer_v2", classes["ISAPNAnalyzer_v2"])

    def test_analyzer_exports_are_explicit(self):
        tree = ast.parse((ROOT / "src/Analyzer/__init__.py").read_text(encoding="utf-8"))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertFalse(
            any(
                isinstance(call.func, ast.Attribute)
                and call.func.attr in {"listdir", "import_module"}
                for call in calls
            )
        )
        assignments = [
            node for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
        ]
        self.assertEqual(len(assignments), 1)

    def test_callable_facade_retains_legacy_namespace(self):
        factory_tree = ast.parse(
            (ROOT / "src/Analyzer/factory.py").read_text(encoding="utf-8")
        )
        analyzer_class = next(
            node
            for node in factory_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Analyzer"
        )
        attributes = {
            target.id
            for node in analyzer_class.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        self.assertTrue(
            {
                "MolAnalyzer",
                "MolAnalyzer_v2",
                "MolAnalyzer_v1p3",
                "ISAAnalyzer",
                "ISAAnalyzer_v2",
                "ISAAnalyzer_v1p3",
                "ISAPNAnalyzer",
                "ISAPNAnalyzer_v2",
                "ISAwSAnalyzer",
                "predict_ensemble",
                "showAtomHighlight",
            }.issubset(attributes)
        )


class AnalyzerCoreHeavyTests(unittest.TestCase):
    def require_runtime(self):
        import importlib.util
        import sys

        required = ("torch", "torch_geometric", "rdkit", "pandas", "numpy", "sklearn", "yaml")
        missing = [name for name in required if importlib.util.find_spec(name) is None]
        if missing:
            self.fail(f"D4CMPP2_RUN_HEAVY=1 but Analyzer dependencies are missing: {missing}")
        package_parent = str(ROOT.parent)
        if package_parent not in sys.path:
            sys.path.insert(0, package_parent)

    @heavy_test
    def test_factory_selects_analyzer_from_saved_contract(self):
        self.require_runtime()
        from D4CMPP2.src.Analyzer.factory import _select_analyzer_class

        cases = (
            (
                {"version": "1.0", "data_manager_class": "MolDataManager"},
                "MolAnalyzer",
            ),
            (
                {"version": "1.3", "data_manager_class": "MolDataManager"},
                "MolAnalyzer_v2",
            ),
            (
                {"version": "2.0", "data_manager_class": "MolDataManager"},
                "MolAnalyzer_v2",
            ),
            (
                {
                    "version": "1.0",
                    "data_manager_class": "ISADataManager",
                    "train_manager_class": "ISATrainer",
                },
                "ISAAnalyzer",
            ),
            (
                {
                    "version": "2.0",
                    "data_manager_class": "ISADataManager",
                    "train_manager_class": "ISATrainer",
                    "molecule_columns": ["compound", "solvent"],
                },
                "ISAAnalyzer_v2",
            ),
            (
                {
                    "version": "2.0",
                    "data_manager_class": "ISADataManager",
                    "train_manager_class": "ISATrainer",
                    "network": "ISATPN_model",
                },
                "ISAPNAnalyzer_v2",
            ),
            (
                {
                    "version": "1.0",
                    "data_manager_class": "ISADataManager",
                    "train_manager_class": "ISATrainer",
                    "network": "ISATPM_model",
                },
                "ISAPNAnalyzer",
            ),
            (
                {
                    "version": "1.0",
                    "data_manager_class": "ISADataManager_withSolv",
                    "network": "ISATwithSolv_model",
                },
                "ISAwSAnalyzer",
            ),
        )
        for config, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(_select_analyzer_class(config).__name__, expected)

    @heavy_test
    def test_structured_result_preserves_duplicates_and_invalid_rows(self):
        self.require_runtime()
        import numpy as np
        from D4CMPP2.src.Analyzer.results import PredictionResult, PredictionRow

        rows = (
            PredictionRow(7, {"compound": "CC"}, np.array([1.0])),
            PredictionRow(8, {"compound": "CC"}, np.array([2.0])),
            PredictionRow(9, {"compound": "invalid"}, None, status="invalid", error="bad molecule"),
        )
        result = PredictionResult(rows, ("Solubility",))
        self.assertEqual([row.row_index for row in result], [7, 8, 9])
        self.assertEqual(len(result.valid_rows), 2)
        self.assertEqual(len(result.invalid_rows), 1)
        frame = result.to_dataframe()
        self.assertEqual(frame["row_index"].tolist(), [7, 8, 9])
        self.assertEqual(frame["prediction_status"].tolist(), ["ok", "ok", "invalid"])
        self.assertTrue(np.isnan(frame.loc[2, "Solubility_pred"]))
        legacy = result.legacy_dict(["compound"])
        self.assertEqual(list(legacy), [("CC",)])
        self.assertEqual(float(legacy[("CC",)][0]), 2.0)

    @heavy_test
    def test_legacy_cache_extensions_are_preserved(self):
        self.require_runtime()
        from D4CMPP2.src.Analyzer.MolAnalyzer import MolAnalyzer

        analyzer = MolAnalyzer.__new__(MolAnalyzer)
        analyzer.data_keys = ["prediction", "fragments"]
        analyzer.for_pickle = ["fragments"]
        self.assertTrue(analyzer.get_file_name("CCO", "prediction").endswith("_0.np"))
        self.assertTrue(analyzer.get_file_name("CCO", "fragments").endswith("_1.pickle"))

    @heavy_test
    def test_input_normalization_is_immutable_and_checks_lengths(self):
        self.require_runtime()
        import numpy as np
        from D4CMPP2.src.Analyzer.core import InferenceCore

        core = InferenceCore.__new__(InferenceCore)
        core.molecule_columns = ("compound", "solvent")
        core.numeric_input_columns = ("temperature",)
        core.input_columns = core.molecule_columns + core.numeric_input_columns

        compound = ["CC", "CCC"]
        solvent = ["O", "O"]
        temperature = [298.0, 310.0]
        normalized = core.normalize_inputs(
            (),
            {
                "compound": compound,
                "solvent": solvent,
                "temperature": temperature,
            },
        )
        normalized["compound"].append("CCCC")
        self.assertEqual(compound, ["CC", "CCC"])
        self.assertEqual(solvent, ["O", "O"])
        self.assertEqual(temperature, [298.0, 310.0])

        with self.assertRaisesRegex(ValueError, "same length"):
            core.normalize_inputs(
                (),
                {
                    "compound": ["CC", "CCC"],
                    "solvent": ["O"],
                    "temperature": [298.0, 310.0],
                },
            )
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            core.normalize_inputs(
                (),
                {
                    "compound": np.array([["CC"], ["CCC"]]),
                    "solvent": ["O", "O"],
                    "temperature": [298.0, 310.0],
                },
            )

    @heavy_test
    def test_prediction_shape_mismatch_is_actionable(self):
        self.require_runtime()
        import numpy as np
        from D4CMPP2.src.Analyzer.core import InferenceCore

        core = InferenceCore.__new__(InferenceCore)
        core.targets = ("a", "b")
        core.input_columns = ("compound",)
        core.dm = type("DM", (), {"graph_errors": []})()
        with self.assertRaisesRegex(
            ValueError,
            r"prediction shape \(1, 1\).*expected \(1, 2\)",
        ):
            core._result_from_scores(
                {"compound": ["CC"]},
                [0],
                np.zeros((1, 1)),
            )

    @heavy_test
    def test_artifact_resolver_reports_all_required_files(self):
        self.require_runtime()
        from D4CMPP2.src.Analyzer.core import resolve_model_artifacts

        with tempfile.TemporaryDirectory(prefix="d4cmpp2-analyzer-artifacts-") as temporary:
            root = Path(temporary) / "model"
            root.mkdir()
            (root / "config.yaml").write_text("target: [y]\n", encoding="utf-8")
            with self.assertRaisesRegex(
                FileNotFoundError,
                r"network\.py.*final\.pth",
            ):
                resolve_model_artifacts(root)

    @heavy_test
    def test_saved_numeric_and_multi_molecule_contract_preserves_rows(self):
        self.require_runtime()
        import torch
        import yaml
        from D4CMPP2 import Analyzer

        with tempfile.TemporaryDirectory(prefix="d4cmpp2-analyzer-inputs-") as temporary:
            root = Path(temporary)
            model = root / "numeric-solvent-model"
            model.mkdir()
            config = {
                "version": "2.0",
                "network": "fixture_numeric",
                "data": "fixture",
                "target": ["property"],
                "target_dim": 1,
                "scaler": "identity",
                "device": "cpu",
                "batch_size": 4,
                "pin_memory": False,
                "num_workers": 0,
                "learning_rate": 0.001,
                "max_epoch": 1,
                "weight_decay": 0.0,
                "optimizer": "Adam",
                "lr_patience": 10,
                "early_stopping_patience": 10,
                "min_lr": 1e-6,
                "molecule_columns": ["compound", "solvent"],
                "numeric_input_columns": ["temperature"],
                "data_manager_module": "MolDataManager",
                "data_manager_class": "MolDataManager",
                "network_manager_module": "NetworkManager",
                "network_manager_class": "NetworkManager",
                "train_manager_module": "TrainManager",
                "train_manager_class": "Trainer",
            }
            (model / "config.yaml").write_text(
                yaml.safe_dump(config, sort_keys=False),
                encoding="utf-8",
            )
            (model / "network.py").write_text(
                "\n".join(
                    [
                        "import torch",
                        "class network(torch.nn.Module):",
                        "    def __init__(self, config):",
                        "        super().__init__()",
                        "        self.bias = torch.nn.Parameter(torch.zeros(1))",
                        "    def forward(self, **batch):",
                        "        return batch['temperature_var'].reshape(-1, 1) + self.bias",
                        "    def loss_fn(self, prediction, target):",
                        "        return torch.mean((prediction - target) ** 2)",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            torch.save({"bias": torch.zeros(1)}, model / "final.pth")

            analyzer = Analyzer(
                model.name,
                model_dir=root,
                save_result=False,
                device="cpu",
            )
            self.assertIsInstance(analyzer, Analyzer.MolAnalyzer_v2)
            with self.assertWarnsRegex(FutureWarning, "MolAnalyzer_v1p3 is deprecated"):
                legacy_analyzer = Analyzer.MolAnalyzer_v1p3(
                    model.name,
                    model_dir=root,
                    save_result=False,
                    device="cpu",
                )
            self.assertIsInstance(legacy_analyzer, Analyzer.MolAnalyzer_v2)
            compound = ["CC", "CC", "CCC"]
            solvent = ["O", "O", "not-a-smiles"]
            temperature = [298.0, 298.0, 310.0]
            result = analyzer.predict_rows(
                compound=compound,
                solvent=solvent,
                temperature=temperature,
            )
            self.assertEqual(compound, ["CC", "CC", "CCC"])
            self.assertEqual(solvent, ["O", "O", "not-a-smiles"])
            self.assertEqual(temperature, [298.0, 298.0, 310.0])
            self.assertEqual([row.status for row in result], ["ok", "ok", "invalid"])
            self.assertEqual(float(result[0].prediction[0]), 298.0)
            self.assertEqual(float(result[1].prediction[0]), 298.0)
            self.assertIn("solvent", result[2].error)

            ensemble = Analyzer.predict_ensemble(
                [analyzer, analyzer],
                compound=["CC", "CCC"],
                solvent=["O", "CO"],
                temperature=[298.0, 310.0],
            )
            self.assertEqual(
                [float(row.prediction[0]) for row in ensemble.mean],
                [298.0, 310.0],
            )
            self.assertEqual(
                [float(row.prediction[0]) for row in ensemble.std],
                [0.0, 0.0],
            )


if __name__ == "__main__":
    unittest.main()
