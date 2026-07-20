import json
import tempfile
import unittest
from pathlib import Path

from markers import heavy_test


class ExperimentLeaderboardTests(unittest.TestCase):
    def _model(self, root, name, metric, target="Solubility", manifests=()):
        import pandas as pd
        import yaml

        model = root / name
        (model / "result").mkdir(parents=True)
        (model / "config.yaml").write_text(
            yaml.safe_dump({
                "network": "GCN",
                "data": "tiny",
                "target": [target],
                "batch_size": 8,
                "learning_rate": 0.001,
            }),
            encoding="utf-8",
        )
        pd.DataFrame(
            {"val_rmse": [metric], "val_r2": [1.0 - metric]},
            index=[target],
        ).to_csv(model / "result" / "metrics.csv")
        for run_id, status, ended_at in manifests:
            path = model / "runs" / run_id / "run_manifest.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({
                "manifest_schema_version": 1,
                "run_id": run_id,
                "status": status,
                "mode": "resume",
                "ended_at": ended_at,
                "network": {"id": "GCN"},
                "config": {"max_epoch": 2},
            }), encoding="utf-8")
        return model

    @heavy_test
    def test_ranks_latest_completed_run_and_includes_legacy_and_failed(self):
        import pandas as pd

        from D4CMPP2 import compare_experiments

        with tempfile.TemporaryDirectory(prefix="d4cmpp2-leaderboard-") as temporary:
            root = Path(temporary)
            self._model(root, "legacy", 0.7)
            self._model(root, "runs", 0.5, manifests=[
                ("old", "completed", "2026-01-01T00:00:00+00:00"),
                ("failed", "failed", "2026-01-02T00:00:00+00:00"),
                ("latest", "completed", "2026-01-03T00:00:00+00:00"),
            ])
            output = root / "comparison" / "leaderboard.csv"
            frame = compare_experiments(root, output_path=output)

            self.assertTrue(output.is_file())
            self.assertEqual(set(frame["status"]), {"legacy", "completed", "failed"})
            latest = frame[frame["run_id"] == "latest"].iloc[0]
            old = frame[frame["run_id"] == "old"].iloc[0]
            failed = frame[frame["run_id"] == "failed"].iloc[0]
            legacy = frame[frame["run_id"] == "legacy"].iloc[0]
            self.assertEqual(latest["rank"], 1)
            self.assertEqual(legacy["rank"], 2)
            self.assertTrue(pd.isna(old["val_rmse"]))
            self.assertTrue(pd.isna(failed["val_rmse"]))
            self.assertIsNone(old["metric_source"])

    @heavy_test
    def test_target_grouping_r2_direction_and_actionable_errors(self):
        import pandas as pd

        from D4CMPP2.src.utils.leaderboard import compare_experiments

        with tempfile.TemporaryDirectory(prefix="d4cmpp2-leaderboard-") as temporary:
            root = Path(temporary)
            self._model(root, "a", 0.2, target="A")
            model_b = self._model(root, "b", 0.8, target="B")
            metrics = pd.read_csv(model_b / "result" / "metrics.csv", index_col=0)
            metrics["val_r2"] = 0.9
            metrics.to_csv(model_b / "result" / "metrics.csv")

            frame = compare_experiments(root, root / "r2.csv", metric="val_r2")
            self.assertEqual(frame.groupby("target")["rank"].min().to_dict(), {"A": 1.0, "B": 1.0})
            with self.assertRaisesRegex(ValueError, "Available targets"):
                compare_experiments(root, root / "missing.csv", target="missing")
            with self.assertRaisesRegex(ValueError, "Available metric columns"):
                compare_experiments(root, root / "missing.csv", metric="val_auc")
            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                compare_experiments(root / "absent")
