import csv
import importlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fixtures import TINY_REGRESSION_CSV, isolated_workdir
from markers import heavy_test


class OptimizeTests(unittest.TestCase):
    def _fake_train(self, **kwargs):
        model_path = Path(kwargs["MODEL_PATH"])
        result = model_path / "result"
        result.mkdir(parents=True)
        score = (
            abs(float(kwargs.get("hidden_dim", 32)) - 24.0) / 10.0
            + float(kwargs.get("dropout", 0.0))
        )
        with open(result / "learning_curve.csv", "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("train_loss", "val_loss"))
            writer.writeheader()
            writer.writerow({"train_loss": score + 1.0, "val_loss": score})
        return str(model_path)

    @heavy_test
    def test_hp_normalization_uses_model_defaults_and_rejects_unknown_keys(self):
        optimize_module = importlib.import_module("D4CMPP2.optimize")

        defaults = optimize_module.normalize_hp("GCN", None, "bayesian")
        self.assertEqual(
            {domain.name for domain in defaults},
            {"hidden_dim", "conv_layers", "linear_layers", "dropout"},
        )
        selected = optimize_module.normalize_hp(
            "GCN",
            {
                "hidden_dim": (16, 64),
                "dropout": [0.0, 0.2],
            },
            "bayesian",
        )
        self.assertEqual(selected[0].low, 16)
        self.assertEqual(selected[0].high, 64)
        self.assertEqual(selected[1].values, (0.0, 0.2))
        with self.assertRaisesRegex(ValueError, "Available keys"):
            optimize_module.normalize_hp("GCN", ["not_defined"], "grid")
        with self.assertRaisesRegex(ValueError, "grid.*bayesian"):
            optimize_module.normalize_hp("GCN", None, "random")
        with self.assertRaisesRegex(ValueError, "positive number"):
            optimize_module.normalize_hp(
                "GCN", {"dropout": {"low": 0.0, "high": 0.5, "step": 0}}, "grid"
            )

    @heavy_test
    def test_grid_returns_best_trial_and_resume_skips_completed_combinations(self):
        optimize_module = importlib.import_module("D4CMPP2.optimize")
        with tempfile.TemporaryDirectory(prefix="d4cmpp2-optimize-grid-") as temp:
            with mock.patch.object(
                optimize_module, "train", side_effect=self._fake_train
            ) as train_mock:
                result = optimize_module.optimize(
                    data="fixture.csv",
                    target=["target"],
                    network="GCN",
                    HP={"hidden_dim": [16, 32], "dropout": [0.0, 0.2]},
                    optimize_strategy="grid",
                    optimization_path=temp,
                )
                self.assertEqual(train_mock.call_count, 4)
                self.assertEqual(result.best_params, {"hidden_dim": 16, "dropout": 0.0})
                self.assertEqual(len(result.trials), 4)
                resumed = optimize_module.optimize(
                    data="fixture.csv",
                    target=["target"],
                    network="GCN",
                    HP={"hidden_dim": [16, 32], "dropout": [0.0, 0.2]},
                    optimize_strategy="grid",
                    optimization_path=temp,
                )
                self.assertEqual(train_mock.call_count, 4)
                self.assertEqual(resumed.best_score, result.best_score)
                self.assertTrue(Path(result.summary_path).exists())
                self.assertTrue(Path(result.summary_path).with_suffix(".csv").exists())
                with self.assertRaisesRegex(ValueError, "different HP search space"):
                    optimize_module.optimize(
                        data="fixture.csv",
                        target=["target"],
                        network="GCN",
                        HP={"hidden_dim": [8]},
                        optimize_strategy="grid",
                        optimization_path=temp,
                    )

    @heavy_test
    def test_bayesian_runs_requested_trials_and_is_reproducible(self):
        optimize_module = importlib.import_module("D4CMPP2.optimize")
        with (
            tempfile.TemporaryDirectory(prefix="d4cmpp2-optimize-bayes-a-") as first,
            tempfile.TemporaryDirectory(prefix="d4cmpp2-optimize-bayes-b-") as second,
        ):
            with mock.patch.object(
                optimize_module, "train", side_effect=self._fake_train
            ) as train_mock:
                result = optimize_module.optimize(
                    data="fixture.csv",
                    target=["target"],
                    network="GCN",
                    HP={
                        "hidden_dim": {"low": 16, "high": 64, "step": 8},
                        "dropout": [0.0, 0.1, 0.2],
                    },
                    optimize_strategy="bayesian",
                    n_trials=6,
                    random_seed=7,
                    optimization_path=first,
                )
                repeated = optimize_module.optimize(
                    data="fixture.csv",
                    target=["target"],
                    network="GCN",
                    HP={
                        "hidden_dim": {"low": 16, "high": 64, "step": 8},
                        "dropout": [0.0, 0.1, 0.2],
                    },
                    optimize_strategy="bayesian",
                    n_trials=6,
                    random_seed=7,
                    optimization_path=second,
                )
        self.assertEqual(train_mock.call_count, 12)
        self.assertEqual(len(result.trials), 6)
        self.assertTrue(all(trial["status"] == "completed" for trial in result.trials))
        self.assertEqual(
            [trial["parameters"] for trial in result.trials],
            [trial["parameters"] for trial in repeated.trials],
        )

    @heavy_test
    def test_one_trial_cpu_training_creates_loadable_model(self):
        optimize_module = importlib.import_module("D4CMPP2.optimize")
        with isolated_workdir() as root:
            (root / "models").mkdir()
            (root / "graphs").mkdir()
            result = optimize_module.optimize(
                data=str(TINY_REGRESSION_CSV),
                target=["target_a"],
                network="GCN",
                HP={"hidden_dim": [8]},
                optimize_strategy="grid",
                optimization_path=root / "optimization",
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

            model_path = Path(result.best_model_path)
            self.assertTrue((model_path / "network.py").is_file())
            self.assertTrue((model_path / "config.yaml").is_file())
            self.assertTrue((model_path / "final.pth").is_file())


if __name__ == "__main__":
    unittest.main()
