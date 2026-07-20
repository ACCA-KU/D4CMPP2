import json
import sys
import tempfile
import unittest
from pathlib import Path

from fixtures import ROOT
from markers import heavy_test


class RunManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if str(ROOT.parent) not in sys.path:
            sys.path.insert(0, str(ROOT.parent))

    @heavy_test
    def test_manifest_records_running_completed_and_failed_states(self):
        from D4CMPP2.src.utils.run_manifest import RunManifest

        with tempfile.TemporaryDirectory(prefix="d4cmpp2-manifest-") as temporary:
            root = Path(temporary)
            data_path = root / "data.csv"
            data_path.write_text("compound,target\nCC,1\n", encoding="utf-8")
            config = {
                "MODEL_PATH": str(root / "model"),
                "DATA_PATH": str(data_path),
                "target": ["target"],
                "network_id": "GCN",
                "network": "GCN_model",
                "data_manager_class": "MolDataManager",
                "network_manager_class": "NetworkManager",
                "train_manager_class": "Trainer",
                "device": "cpu",
            }
            manifest = RunManifest(config, "scratch")
            running = json.loads(manifest.path.read_text(encoding="utf-8"))
            self.assertEqual(running["status"], "running")
            self.assertEqual(running["mode"], "scratch")
            self.assertEqual(len(running["data"]["sha256"]), 64)

            manifest.finish("completed", completed_epoch=2)
            completed = json.loads(manifest.path.read_text(encoding="utf-8"))
            self.assertEqual(completed["status"], "completed")
            self.assertEqual(completed["completed_epoch"], 2)
            self.assertIn("duration_seconds", completed)

            failed = RunManifest(config, "resume")
            error = ValueError("bounded failure")
            failed.finish("failed", error=error)
            payload = json.loads(failed.path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "failed")
            self.assertEqual(payload["error"]["type"], "ValueError")
            self.assertEqual(payload["error"]["message"], "bounded failure")


if __name__ == "__main__":
    unittest.main()
