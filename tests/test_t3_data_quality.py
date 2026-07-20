import json
import tempfile
import unittest
from pathlib import Path

from markers import heavy_test


class DataQualityReportTests(unittest.TestCase):
    @heavy_test
    def test_reports_duplicates_conflicts_missing_invalid_and_split_overlap(self):
        import pandas as pd

        from D4CMPP2.src.utils.data_quality import write_data_quality_report

        with tempfile.TemporaryDirectory(prefix="d4cmpp2-quality-") as temporary:
            root = Path(temporary)
            frame = pd.DataFrame({
                "compound": ["CCO", "OCC", "invalid", ""],
                "target": [1.0, 2.0, None, 4.0],
                "set": ["train", "val", "train", "test"],
            })
            original = frame.copy(deep=True)
            config = {
                "MODEL_PATH": str(root / "model"),
                "DATA_PATH": str(root / "data.csv"),
                "molecule_columns": ["compound"],
                "target": ["target"],
            }
            report_path, issues_path = write_data_quality_report(
                frame,
                config,
                graph_errors=[{
                    "row_index": 2,
                    "type": "compound",
                    "smiles": "invalid",
                    "reason": "parse failed",
                }],
            )

            pd.testing.assert_frame_equal(frame, original)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            issues = pd.read_csv(issues_path)
            self.assertEqual(report["row_count"], 4)
            self.assertEqual(report["target_summary"]["target"]["missing_count"], 1)
            expected = {
                "duplicate_molecule",
                "conflicting_duplicate_target",
                "split_molecule_overlap",
                "invalid_smiles",
                "empty_smiles",
                "missing_target",
                "graph_generation_failure",
            }
            self.assertTrue(expected.issubset(set(issues["issue"])))

    @heavy_test
    def test_manager_writes_report_before_strict_validation_failure(self):
        import pandas as pd

        from D4CMPP2.src.DataManager.MolDataManager import MolDataManager

        with tempfile.TemporaryDirectory(prefix="d4cmpp2-quality-") as temporary:
            root = Path(temporary)
            data_path = root / "invalid.csv"
            pd.DataFrame({
                "compound": ["CC", "CCC"],
                "target": ["bad", 1.0],
                "set": ["train", "val"],
            }).to_csv(data_path, index=False)
            manager = object.__new__(MolDataManager)
            manager.config = {
                "MODEL_PATH": str(root / "model"),
                "DATA_PATH": str(data_path),
                "data_quality_report": True,
                "molecule_columns": ["compound"],
                "target": ["target"],
            }
            manager.molecule_columns = ["compound"]
            manager.target = ["target"]
            manager.numeric_input_columns = []
            manager.graph_errors = []

            with self.assertRaisesRegex(ValueError, "non-numeric"):
                manager.load_data()
            report_path = root / "model" / "data_quality_report.json"
            issues_path = root / "model" / "data_quality_issues.csv"
            self.assertTrue(report_path.is_file())
            self.assertIn("missing_target", pd.read_csv(issues_path)["issue"].tolist())
