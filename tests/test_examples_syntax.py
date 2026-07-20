import json
from pathlib import Path
import unittest

from fixtures import ROOT


class ExampleSyntaxTests(unittest.TestCase):
    EXPECTED_GUIDES = (
        "training/01_basic_cpu.py",
        "training/02_solvent.py",
        "training/03_multitarget.py",
        "training/04_splitting_reproducibility.py",
        "training/05_isa.py",
        "training/06_saved_model_modes.py",
        "training/07_model_families.py",
        "inference/01_prediction.py",
        "inference/02_csv_prediction.py",
        "inference/03_uncertainty_ensemble.py",
        "inference/04_isa_interpretation.py",
        "experiments/01_compare.py",
        "experiments/02_optimize.py",
        "experiments/03_legacy_grid_search.py",
        "extensions/01_callbacks.py",
        "extensions/02_custom_network_training.py",
        "extensions/03_numeric_inputs.py",
        "cli/README.md",
    )

    def test_python_examples_compile(self):
        for path in (ROOT / "examples").rglob("*.py"):
            with self.subTest(path=path.relative_to(ROOT)):
                compile(path.read_text(encoding="utf-8"), str(path), "exec")

    def test_notebook_code_cells_compile_and_use_current_package(self):
        notebooks = list((ROOT / "examples").rglob("*.ipynb"))
        self.assertTrue(notebooks)
        for path in notebooks:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            code = "\n\n".join(
                "".join(cell.get("source", []))
                for cell in notebook.get("cells", [])
                if cell.get("cell_type") == "code"
            )
            with self.subTest(path=path.relative_to(ROOT)):
                compile(code, str(path), "exec")
                self.assertNotIn("from D4CMPP import", code)
                self.assertNotIn("conv_layer=", code)

    def test_feature_guide_indexes_every_supported_example(self):
        examples = ROOT / "examples"
        guide = (examples / "README.md").read_text(encoding="utf-8")
        for relative in self.EXPECTED_GUIDES:
            with self.subTest(example=relative):
                self.assertTrue((examples / relative).is_file())
                self.assertIn(relative, guide)
        self.assertTrue((examples / "assets/tiny_numeric.csv").is_file())
        self.assertIn("examples/README.md", (ROOT / "README.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
