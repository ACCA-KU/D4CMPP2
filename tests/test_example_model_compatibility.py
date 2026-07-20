import hashlib
import shutil
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from fixtures import ROOT
from markers import heavy_test


EXAMPLE_MODELS = {
    "GCN": {
        "path": ROOT
        / "examples/general/assets/models/GCN_model_Aqsoldb_Solubility_20240101_000000",
        "parameters": 8625,
        "sha256": "968d10acc3f2f6b0f988199942c264446e5715c248f18904b7553e820b87e475",
    },
    "ISAT": {
        "path": ROOT
        / "examples/ISA/assets/Models/ISAT_model_Aqsoldb_Solubility_620_20240101_000000",
        "parameters": 439620,
        "sha256": "5685d0952ee175898c390c3c589b836d65bf939ceda892f9d2bec12ee48f3986",
    },
    "ISATPM": {
        "path": ROOT
        / "examples/ISA/assets/Models/ISATPM_model_Aqsoldb_Solubility_620_20240101_000000",
        "parameters": 567683,
        "sha256": "e2205ffa5e3fe540edf0e714714daa1755d691129ad81c47ca4e4eabe9d315cd",
    },
}


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def prevent_bytecode_writes():
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        yield
    finally:
        sys.dont_write_bytecode = previous


class ExampleModelCompatibilityTests(unittest.TestCase):
    def require_runtime(self):
        import importlib.util
        import sys

        required = ("torch", "torch_geometric", "rdkit", "pandas", "numpy", "sklearn", "yaml")
        missing = [name for name in required if importlib.util.find_spec(name) is None]
        if missing:
            self.fail(f"D4CMPP2_RUN_HEAVY=1 but example-model dependencies are missing: {missing}")

        package_parent = str(ROOT.parent)
        if package_parent not in sys.path:
            sys.path.insert(0, package_parent)

    @heavy_test
    def test_v2_factory_loads_isa_contract_and_deprecated_name(self):
        self.require_runtime()

        import yaml
        from D4CMPP2 import Analyzer

        source = EXAMPLE_MODELS["ISAT"]["path"]
        with tempfile.TemporaryDirectory(prefix="d4cmpp2-isa-v2-") as temporary:
            model = Path(temporary) / "isat-v2"
            model.mkdir()
            for name in (
                "config.yaml",
                "network.py",
                "final.pth",
                "scaler.pkl",
                "functional_group.csv",
            ):
                shutil.copy2(source / name, model / name)

            with (model / "config.yaml").open(encoding="utf-8") as file:
                config = yaml.load(file, Loader=yaml.FullLoader)
            config["version"] = "2.0"
            with (model / "config.yaml").open("w", encoding="utf-8") as file:
                yaml.dump(config, file)

            analyzer = Analyzer(model, save_result=False, device="cpu")
            self.assertIsInstance(analyzer, Analyzer.ISAAnalyzer_v2)
            result = analyzer.analyze_rows(["CCO"], include_features=False)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].score_mode, "atom")

            with self.assertWarnsRegex(FutureWarning, "ISAAnalyzer_v1p3 is deprecated"):
                legacy_name = Analyzer.ISAAnalyzer_v1p3(
                    model,
                    save_result=False,
                    device="cpu",
                )
            self.assertIsInstance(legacy_name, Analyzer.ISAAnalyzer_v2)

    @heavy_test
    def test_example_weight_snapshots_load_read_only(self):
        """Load each original network snapshot and weight file without copying the binary."""
        self.require_runtime()

        import yaml

        from D4CMPP2.src.utils import module_loader

        for model_name, fixture in EXAMPLE_MODELS.items():
            model_path = fixture["path"]
            weight_path = model_path / "final.pth"
            before = file_sha256(weight_path)
            with self.subTest(model=model_name):
                self.assertEqual(before, fixture["sha256"])
                with (model_path / "config.yaml").open(encoding="utf-8") as file:
                    config = yaml.load(file, Loader=yaml.FullLoader)
                config["MODEL_PATH"] = str(model_path)
                config["LOAD_PATH"] = str(model_path)
                config["device"] = "cpu"

                with prevent_bytecode_writes():
                    manager = module_loader.load_network_manager(config)(config, unwrapper=None, temp=True)

                parameter_count = sum(parameter.numel() for parameter in manager.network.parameters())
                self.assertEqual(parameter_count, fixture["parameters"])
                self.assertEqual(file_sha256(weight_path), before)

    @heavy_test
    def test_legacy_example_analyzers_support_cpu_and_aligned_isa_results(self):
        self.require_runtime()

        from D4CMPP2 import Analyzer

        gcn = Analyzer(
            EXAMPLE_MODELS["GCN"]["path"],
            save_result=False,
            device="cpu",
        )
        self.assertIsInstance(gcn, Analyzer.MolAnalyzer)
        gcn_result = gcn.predict_rows(compound=["CCO", "CCO", "not-a-smiles"])
        self.assertEqual([row.status for row in gcn_result], ["ok", "ok", "invalid"])
        self.assertEqual(gcn_result[0].prediction.shape, (1,))
        self.assertEqual(gcn_result[1].prediction.shape, (1,))

        isat = Analyzer(
            EXAMPLE_MODELS["ISAT"]["path"],
            save_result=False,
            device="cpu",
        )
        self.assertIsInstance(isat, Analyzer.ISAAnalyzer)
        isat_result = isat.analyze_rows(["CCO"], include_features=False)
        self.assertEqual(len(isat_result), 1)
        self.assertEqual(isat_result[0].score_mode, "atom")
        self.assertEqual(
            isat_result[0].scores["positive"].shape[0],
            isat_result[0].atom_count,
        )
        self.assertNotIn("negative", isat_result[0].scores)
        self.assertEqual(isat_result[0].features, {})
        self.assertEqual(
            sorted(
                atom
                for fragment in isat_result[0].fragment_atom_indices
                for atom in fragment
            ),
            list(range(isat_result[0].atom_count)),
        )

        isatpm = Analyzer(
            EXAMPLE_MODELS["ISATPM"]["path"],
            save_result=False,
            device="cpu",
        )
        self.assertIsInstance(isatpm, Analyzer.ISAPNAnalyzer)
        isatpm_result = isatpm.analyze_rows(["CCO"], include_features=True)
        self.assertEqual(len(isatpm_result), 1)
        self.assertEqual(isatpm_result[0].score_mode, "fragment")
        self.assertEqual(
            isatpm_result[0].scores["positive"].shape[0],
            len(isatpm_result[0].fragment_atom_indices),
        )
        self.assertIn("negative", isatpm_result[0].scores)
        self.assertEqual(isatpm_result[0].feature_mode, "fragment")
        self.assertEqual(
            isatpm_result[0].features["feature_P"].shape[0],
            len(isatpm_result[0].fragment_atom_indices),
        )
        self.assertIn("feature_N", isatpm_result[0].features)
        for fixture in EXAMPLE_MODELS.values():
            self.assertFalse(
                (fixture["path"] / "__pycache__").exists(),
                "Analyzer must not write bytecode into a saved model folder",
            )

if __name__ == "__main__":
    unittest.main()
