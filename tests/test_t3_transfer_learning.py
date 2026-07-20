import json
import tempfile
import unittest
from pathlib import Path

from markers import heavy_test


NETWORK_SOURCE = """\
import torch

class network(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2)
        self.head = torch.nn.Linear(2, config["target_dim"])
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, features, **kwargs):
        return self.head(self.backbone(features))
"""


class TransferLearningTests(unittest.TestCase):
    @heavy_test
    def test_transfer_loads_compatible_backbone_and_reports_skipped_head(self):
        import torch

        from D4CMPP2.src.NetworkManager.NetworkManager import NetworkManager

        with tempfile.TemporaryDirectory(prefix="d4cmpp2-transfer-") as temporary:
            root = Path(temporary)
            source = root / "source"
            target = root / "target"
            source.mkdir()
            target.mkdir()
            (source / "network.py").write_text(NETWORK_SOURCE, encoding="utf-8")
            (target / "network.py").write_text(NETWORK_SOURCE, encoding="utf-8")
            (source / "config.yaml").write_text(
                "target_dim: 1\nnetwork: fixture_source\n",
                encoding="utf-8",
            )

            namespace = {}
            exec(NETWORK_SOURCE, namespace)
            source_network = namespace["network"]({"target_dim": 1})
            with torch.no_grad():
                source_network.backbone.weight.fill_(7.0)
                source_network.backbone.bias.fill_(3.0)
                source_network.head.weight.fill_(11.0)
                source_network.head.bias.fill_(13.0)
            torch.save(source_network.state_dict(), source / "final.pth")

            config = {
                "MODEL_PATH": str(target),
                "target_dim": 2,
                "network": "fixture_target",
                "network_id": "fixture_target",
                "device": "cpu",
                "pin_memory": False,
                "optimizer": "Adam",
                "learning_rate": 0.001,
                "weight_decay": 0.0,
            }
            manager = NetworkManager(
                config,
                tf_path=str(source),
                unwrapper=lambda **batch: batch,
            )

            self.assertTrue(
                torch.equal(
                    manager.network.backbone.weight,
                    source_network.backbone.weight,
                )
            )
            self.assertTrue(
                torch.equal(
                    manager.network.backbone.bias,
                    source_network.backbone.bias,
                )
            )
            report_path = target / "transfer_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(
                report["counts"],
                {
                    "loaded": 2,
                    "shape_mismatch": 2,
                    "source_only": 0,
                    "target_only": 0,
                },
            )
            self.assertEqual(
                {item["name"] for item in report["loaded"]},
                {"backbone.weight", "backbone.bias"},
            )
            self.assertEqual(
                {item["name"] for item in report["shape_mismatch"]},
                {"head.weight", "head.bias"},
            )
            self.assertEqual(len(report["source_weights_sha256"]), 64)

    @heavy_test
    def test_lr_dict_rejects_unmatched_and_ambiguous_components(self):
        import torch

        from D4CMPP2.src.NetworkManager.NetworkManager import NetworkManager

        manager = object.__new__(NetworkManager)
        manager.config = {
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
        }
        manager.network = torch.nn.Sequential(
            torch.nn.Sequential(torch.nn.Linear(2, 2))
        )
        with self.assertRaisesRegex(ValueError, "did not match"):
            manager.init_optimizer({"missing": 0.01})
        with self.assertRaisesRegex(ValueError, "both match"):
            manager.init_optimizer({"0": 0.01, "weight": 0.02})

    @heavy_test
    def test_incompatible_source_snapshot_error_is_actionable(self):
        import torch

        from D4CMPP2.src.NetworkManager.NetworkManager import NetworkManager

        with tempfile.TemporaryDirectory(prefix="d4cmpp2-transfer-bad-") as temporary:
            source = Path(temporary)
            (source / "network.py").write_text(NETWORK_SOURCE, encoding="utf-8")
            (source / "config.yaml").write_text(
                "target_dim: 1\nnetwork: fixture\n",
                encoding="utf-8",
            )
            torch.save({"wrong.weight": torch.ones(1)}, source / "final.pth")

            manager = object.__new__(NetworkManager)
            manager.device = "cpu"
            manager.config = {"MODEL_PATH": str(source), "network": "target"}
            manager.network = torch.nn.Linear(2, 1)
            with self.assertRaisesRegex(
                RuntimeError,
                "incompatible with its saved network.py/config.yaml snapshot",
            ):
                manager.load_params_transfer_learn(str(source))


if __name__ == "__main__":
    unittest.main()
