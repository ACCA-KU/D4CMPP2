import copy
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fixtures import TINY_REGRESSION_CSV, isolated_workdir
from markers import heavy_test


class GridSearchIsolationTests(unittest.TestCase):
    @heavy_test
    def test_calls_are_isolated_and_summary_records_success_and_failure(self):
        grid_module = importlib.import_module("D4CMPP2.grid_search")

        original_defaults = copy.deepcopy(grid_module.config0)
        with tempfile.TemporaryDirectory(prefix="d4cmpp2-grid-") as temporary:
            root = Path(temporary)
            base_paths = [root / "first", root / "second"]
            seen_configs = []

            def fake_set_config(**config):
                output = copy.deepcopy(config)
                output["MODEL_PATH"] = str(base_paths.pop(0))
                return output

            def fake_run(config):
                seen_configs.append(copy.deepcopy(config))
                if config["hidden_dim"] == 64:
                    raise RuntimeError("intentional trial failure")
                return config["MODEL_PATH"]

            with (
                mock.patch.object(
                    grid_module,
                    "check_args",
                    side_effect=lambda **values: values,
                ),
                mock.patch.object(grid_module, "set_config", side_effect=fake_set_config),
                mock.patch.object(grid_module, "run", side_effect=fake_run),
                mock.patch.object(
                    grid_module.supportfile_saver,
                    "save_additional_files",
                ),
            ):
                self.assertIsNone(
                    grid_module.grid_search(
                        {"hidden_dim": [32, 64]},
                        data="first.csv",
                        network="GCN",
                    )
                )
                self.assertIsNone(
                    grid_module.grid_search(
                        {"hidden_dim": [16]},
                        data="second.csv",
                        network="GCN",
                    )
                )

            self.assertEqual(grid_module.config0, original_defaults)
            self.assertEqual(
                [config["data"] for config in seen_configs],
                ["first.csv", "first.csv", "second.csv"],
            )
            first_summary = json.loads(
                (root / "first_grid_search.json").read_text(encoding="utf-8")
            )
            self.assertEqual(first_summary["status"], "completed_with_failures")
            self.assertEqual(first_summary["completed_count"], 1)
            self.assertEqual(first_summary["failed_count"], 1)
            self.assertEqual(
                [trial["status"] for trial in first_summary["trials"]],
                ["completed", "failed"],
            )
            self.assertEqual(
                first_summary["trials"][1]["error_type"], "RuntimeError"
            )
            self.assertTrue((root / "first_grid_search.csv").is_file())

    @heavy_test
    def test_duplicate_combinations_get_distinct_paths_and_support_files(self):
        grid_module = importlib.import_module("D4CMPP2.grid_search")

        with tempfile.TemporaryDirectory(prefix="d4cmpp2-grid-duplicate-") as temporary:
            root = Path(temporary)
            base = root / "model"
            paths = []
            support_paths = []

            def fake_set_config(**config):
                return {**copy.deepcopy(config), "MODEL_PATH": str(base)}

            def fake_run(config):
                paths.append(config["MODEL_PATH"])
                return config["MODEL_PATH"]

            def fake_support(config):
                support_paths.append(config["MODEL_PATH"])

            with (
                mock.patch.object(
                    grid_module,
                    "check_args",
                    side_effect=lambda **values: values,
                ),
                mock.patch.object(grid_module, "set_config", side_effect=fake_set_config),
                mock.patch.object(grid_module, "run", side_effect=fake_run),
                mock.patch.object(
                    grid_module.supportfile_saver,
                    "save_additional_files",
                    side_effect=fake_support,
                ),
            ):
                grid_module.grid_search({"dropout": [0.1, 0.1]})

            self.assertEqual(len(set(paths)), 2)
            self.assertEqual(paths, support_paths)
            self.assertEqual(paths[0], f"{base}_dropout,0.1")
            self.assertIn("__trial_0002", paths[1])

    @heavy_test
    def test_invalid_grid_is_actionable_before_configuration(self):
        grid_module = importlib.import_module("D4CMPP2.grid_search")

        with mock.patch.object(grid_module, "set_config") as set_config:
            with self.assertRaisesRegex(ValueError, "non-empty mapping"):
                grid_module.grid_search({})
            with self.assertRaisesRegex(ValueError, "invalid entries"):
                grid_module.grid_search({"hidden_dim": []})
            set_config.assert_not_called()

    @heavy_test
    def test_one_trial_cpu_training_creates_loadable_model_and_summary(self):
        grid_module = importlib.import_module("D4CMPP2.grid_search")

        with isolated_workdir() as root:
            (root / "models").mkdir()
            (root / "graphs").mkdir()
            base = root / "grid_model"
            result = grid_module.grid_search(
                {"hidden_dim": [8]},
                data=str(TINY_REGRESSION_CSV),
                target=["target_a"],
                network="GCN",
                MODEL_PATH=str(base),
                MODEL_DIR=str(root / "models"),
                GRAPH_DIR=str(root / "graphs"),
                device="cpu",
                pin_memory=False,
                random_seed=123,
                max_epoch=1,
                batch_size=4,
                conv_layers=2,
                linear_layers=2,
                dropout=0.1,
                save_prediction=False,
            )

            self.assertIsNone(result)
            summary = json.loads(
                (root / "grid_model_grid_search.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["completed_count"], 1)
            trial_path = Path(summary["trials"][0]["model_path"])
            self.assertTrue((trial_path / "network.py").is_file())
            self.assertTrue((trial_path / "config.yaml").is_file())
            self.assertTrue((trial_path / "final.pth").is_file())


if __name__ == "__main__":
    unittest.main()
