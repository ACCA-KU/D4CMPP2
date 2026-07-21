import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from fixtures import ROOT
from markers import heavy_test


SOLVENT_NETWORK_IDS = ("GCNwS", "MPNNwS", "DMPNNwS", "AFPwS", "GATwS")


class SolventAnalyzerEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if str(ROOT.parent) not in sys.path:
            sys.path.insert(0, str(ROOT.parent))

    @heavy_test
    def test_all_solvent_models_train_save_reload_and_predict(self):
        script = ROOT / "examples" / "integration" / "solvent_analyzer.py"
        with tempfile.TemporaryDirectory(prefix="d4cmpp2-solvent-example-") as temporary:
            root = Path(temporary)
            output_root = root / "output"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--output-root",
                    str(output_root),
                ],
                cwd=root,
                env=os.environ.copy(),
                text=True,
                capture_output=True,
                timeout=180,
                check=False,
            )
            self.assertEqual(
                completed.returncode,
                0,
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
            )
            for network_id in SOLVENT_NETWORK_IDS:
                model_path = output_root / f"saved-{network_id}"
                for artifact in ("network.py", "config.yaml", "final.pth", "scaler.pkl"):
                    self.assertTrue((model_path / artifact).is_file())
                self.assertIn(f'"{network_id}"', completed.stdout)


if __name__ == "__main__":
    unittest.main()
