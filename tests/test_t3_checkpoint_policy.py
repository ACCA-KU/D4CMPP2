import sys
import tempfile
import unittest
from pathlib import Path

from fixtures import ROOT
from markers import heavy_test


class CheckpointPolicyCharacterizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if str(ROOT.parent) not in sys.path:
            sys.path.insert(0, str(ROOT.parent))

    @heavy_test
    def test_both_schedulers_step_and_lr_reduction_restores_weights_only(self):
        import torch

        from D4CMPP2.src.NetworkManager.NetworkManager import NetworkManager

        with tempfile.TemporaryDirectory(prefix="d4cmpp2-scheduler-policy-") as temporary:
            manager = object.__new__(NetworkManager)
            manager.config = {"MODEL_PATH": temporary}
            manager.device = "cpu"
            manager.network = torch.nn.Linear(1, 1, bias=False)
            manager.optimizer = torch.optim.SGD(
                manager.network.parameters(), lr=1.0, momentum=0.9
            )
            manager.schedulers = [
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    manager.optimizer, patience=0, factor=0.1, min_lr=1e-7
                ),
                torch.optim.lr_scheduler.StepLR(
                    manager.optimizer, step_size=2, gamma=0.5
                ),
            ]
            manager.best_loss = float("inf")
            manager.es_patience = 10
            manager.es_counter = 0
            manager.last_lr = 1.0

            with torch.no_grad():
                manager.network.weight.fill_(1.0)
            manager.optimizer.zero_grad()
            manager.network.weight.grad = torch.zeros_like(manager.network.weight)
            manager.optimizer.step()
            self.assertIsNone(manager.scheduler_step(1.0))
            self.assertEqual(manager.best_loss, 1.0)
            self.assertTrue((Path(temporary) / "param_1.0.pth").is_file())

            manager.network.weight.grad = torch.ones_like(manager.network.weight)
            manager.optimizer.step()
            momentum_before = manager.optimizer.state[manager.network.weight][
                "momentum_buffer"
            ].clone()
            with torch.no_grad():
                manager.network.weight.fill_(9.0)

            self.assertIsNone(manager.scheduler_step(2.0))
            self.assertEqual(manager.schedulers[0].last_epoch, 2)
            self.assertEqual(manager.schedulers[1].last_epoch, 2)
            self.assertAlmostEqual(manager.get_lr(), 0.05)
            self.assertAlmostEqual(manager.network.weight.item(), 1.0)
            self.assertTrue(
                torch.equal(
                    manager.optimizer.state[manager.network.weight]["momentum_buffer"],
                    momentum_before,
                )
            )

    @heavy_test
    def test_early_stopping_uses_greater_than_patience_and_best_weights(self):
        import torch

        from D4CMPP2.src.NetworkManager.NetworkManager import NetworkManager

        with tempfile.TemporaryDirectory(prefix="d4cmpp2-early-stop-policy-") as temporary:
            manager = object.__new__(NetworkManager)
            manager.config = {"MODEL_PATH": temporary}
            manager.device = "cpu"
            manager.network = torch.nn.Linear(1, 1, bias=False)
            manager.optimizer = torch.optim.SGD(manager.network.parameters(), lr=0.1)
            manager.schedulers = [
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    manager.optimizer, patience=100
                ),
                torch.optim.lr_scheduler.StepLR(
                    manager.optimizer, step_size=100, gamma=0.9
                ),
            ]
            manager.best_loss = float("inf")
            manager.es_patience = 1
            manager.es_counter = 0
            manager.last_lr = 0.1

            with torch.no_grad():
                manager.network.weight.fill_(2.0)
            manager.optimizer.zero_grad()
            manager.network.weight.grad = torch.zeros_like(manager.network.weight)
            manager.optimizer.step()
            self.assertIsNone(manager.scheduler_step(1.0))

            with torch.no_grad():
                manager.network.weight.fill_(7.0)
            self.assertIsNone(manager.scheduler_step(2.0))
            self.assertEqual(manager.es_counter, 1)
            self.assertTrue(manager.scheduler_step(3.0))
            self.assertEqual(manager.es_counter, 2)
            self.assertAlmostEqual(manager.network.weight.item(), 2.0)

    @heavy_test
    def test_load_path_continue_restores_final_weights_and_curve_but_resets_state(self):
        import pandas as pd
        import torch

        from D4CMPP2.src.NetworkManager.NetworkManager import NetworkManager

        network_source = """
import torch

class network(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1, bias=False)

    def forward(self, **kwargs):
        return self.linear(kwargs["value"])

    def loss_fn(self, scores, targets):
        return torch.nn.functional.mse_loss(scores, targets)
"""
        with tempfile.TemporaryDirectory(prefix="d4cmpp2-continue-policy-") as temporary:
            model_path = Path(temporary)
            (model_path / "network.py").write_text(network_source, encoding="utf-8")
            config = {
                "MODEL_PATH": str(model_path),
                "device": "cpu",
                "pin_memory": False,
                "optimizer": "Adam",
                "learning_rate": 0.01,
                "weight_decay": 0.0,
                "lr_patience": 2,
                "min_lr": 1e-7,
                "lr_plateau_decay": 0.1,
                "lr_step": 3,
                "lr_step_decay": 0.9,
            }

            first = NetworkManager(dict(config), unwrapper=lambda **batch: batch)
            with torch.no_grad():
                first.network.linear.weight.fill_(2.5)
            torch.save(first.network.state_dict(), model_path / "final.pth")
            result_path = model_path / "result"
            result_path.mkdir()
            pd.DataFrame(
                [{"train_loss": 1.5, "val_loss": 2.5}]
            ).to_csv(result_path / "learning_curve.csv", index=False)

            continued = NetworkManager(dict(config), unwrapper=lambda **batch: batch)
            self.assertAlmostEqual(continued.network.linear.weight.item(), 2.5)
            self.assertEqual(continued.learning_curve.shape, (1, 2))
            self.assertEqual(continued.optimizer.state, {})
            self.assertEqual(continued.best_loss, float("inf"))
            self.assertEqual(continued.es_counter, 0)
            self.assertEqual(continued.last_lr, 0.01)
            self.assertEqual(len(continued.schedulers), 2)

    @heavy_test
    def test_full_resume_matches_uninterrupted_training_state(self):
        import random

        import numpy as np
        import torch

        from D4CMPP2.src.NetworkManager.NetworkManager import NetworkManager

        def make_manager(path):
            path.mkdir(parents=True, exist_ok=True)
            manager = object.__new__(NetworkManager)
            manager.config = {
                "MODEL_PATH": str(path),
                "network_id": "fixture",
                "network": "fixture_network",
                "target": ["value"],
                "target_dim": 1,
                "optimizer": "Adam",
                "scheduler_policy": "legacy_dual",
            }
            manager.device = "cpu"
            manager.network = torch.nn.Linear(2, 1)
            manager.optimizer = torch.optim.Adam(manager.network.parameters(), lr=0.01)
            manager.schedulers = [
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    manager.optimizer, patience=2, factor=0.5
                ),
                torch.optim.lr_scheduler.StepLR(
                    manager.optimizer, step_size=3, gamma=0.9
                ),
            ]
            manager.best_loss = float("inf")
            manager.best_epoch = None
            manager.es_patience = 20
            manager.es_counter = 0
            manager.last_lr = 0.01
            manager.completed_epoch = -1
            manager.next_epoch = 0
            manager.run_id = "fixture-run"
            return manager

        def run_epochs(manager, count):
            x = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
            y = torch.tensor([[0.5], [-0.25], [0.75]])
            for epoch in range(manager.next_epoch, manager.next_epoch + count):
                factor = 1.0 + 0.01 * random.random() + 0.01 * float(np.random.random())
                jitter = 0.001 * torch.rand_like(x)
                manager.optimizer.zero_grad()
                prediction = manager.network((x + jitter) * factor)
                loss = torch.nn.functional.mse_loss(prediction, y)
                loss.backward()
                manager.optimizer.step()
                manager.scheduler_step(float(loss.detach()), completed_epoch=epoch)

        def optimizer_tensors(manager):
            return [
                value.detach().clone()
                for state in manager.optimizer.state.values()
                for value in state.values()
                if torch.is_tensor(value)
            ]

        with tempfile.TemporaryDirectory(prefix="d4cmpp2-exact-resume-") as temporary:
            root = Path(temporary)
            full_path = root / "full"
            split_path = root / "split"

            random.seed(123)
            np.random.seed(123)
            torch.manual_seed(123)
            full = make_manager(full_path)
            run_epochs(full, 4)

            random.seed(123)
            np.random.seed(123)
            torch.manual_seed(123)
            first = make_manager(split_path)
            run_epochs(first, 2)

            resumed = make_manager(split_path)
            resumed.load_training_checkpoint(split_path)
            self.assertEqual(resumed.next_epoch, 2)
            run_epochs(resumed, 2)

            self.assertEqual(resumed.next_epoch, full.next_epoch)
            self.assertEqual(resumed.best_epoch, full.best_epoch)
            self.assertAlmostEqual(resumed.best_loss, full.best_loss, places=12)
            self.assertEqual(resumed.es_counter, full.es_counter)
            self.assertAlmostEqual(resumed.get_lr(), full.get_lr(), places=15)
            for observed, expected in zip(
                resumed.network.parameters(), full.network.parameters()
            ):
                self.assertTrue(torch.equal(observed, expected))
            for observed, expected in zip(
                optimizer_tensors(resumed), optimizer_tensors(full)
            ):
                self.assertTrue(torch.equal(observed, expected))
            self.assertEqual(
                [scheduler.state_dict() for scheduler in resumed.schedulers],
                [scheduler.state_dict() for scheduler in full.schedulers],
            )

    @heavy_test
    def test_resume_schema_mismatch_is_actionable(self):
        import torch

        from D4CMPP2.src.NetworkManager.NetworkManager import NetworkManager

        with tempfile.TemporaryDirectory(prefix="d4cmpp2-resume-mismatch-") as temporary:
            manager = object.__new__(NetworkManager)
            manager.config = {
                "MODEL_PATH": temporary,
                "network_id": "fixture",
                "network": "fixture_network",
                "target": ["value"],
                "target_dim": 1,
                "optimizer": "SGD",
                "scheduler_policy": "legacy_dual",
            }
            manager.device = "cpu"
            manager.network = torch.nn.Linear(1, 1)
            manager.optimizer = torch.optim.SGD(manager.network.parameters(), lr=0.1)
            manager.schedulers = [
                torch.optim.lr_scheduler.ReduceLROnPlateau(manager.optimizer),
                torch.optim.lr_scheduler.StepLR(manager.optimizer, step_size=2),
            ]
            manager.best_loss = 1.0
            manager.best_epoch = 0
            manager.es_counter = 0
            manager.last_lr = 0.1
            manager.run_id = "fixture"
            manager.save_training_checkpoint("latest", 0)
            manager.config["target_dim"] = 2
            with self.assertRaisesRegex(
                ValueError,
                "incompatible for target_dim.*checkpoint=1.*current=2.*LOAD_PATH.*TRANSFER_PATH",
            ):
                manager.load_training_checkpoint(temporary)


if __name__ == "__main__":
    unittest.main()
