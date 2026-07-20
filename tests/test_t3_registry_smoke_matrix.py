import importlib
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from fixtures import ROOT
from markers import heavy_test


EXPECTED_IDS = {
    "GCN",
    "GCNwS",
    "MPNN",
    "MPNNwS",
    "DMPNN",
    "DMPNNwS",
    "AFP",
    "AFPwS",
    "GAT",
    "GATwS",
    "GC",
    "ISAT",
    "ISATPN",
}


class RegistrySmokeMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if str(ROOT.parent) not in sys.path:
            sys.path.insert(0, str(ROOT.parent))

    @staticmethod
    def _base_config(model_path):
        return {
            "data": "registry_smoke",
            "target": ["value"],
            "target_dim": 1,
            "MODEL_PATH": str(model_path),
            "device": "cpu",
            "pin_memory": False,
            "molecule_columns": ["compound"],
            "numeric_input_columns": [],
            "scaler": "identity",
            "split_random_seed": 42,
            "hidden_dim": 8,
            "conv_layers": 1,
            "linear_layers": 1,
            "dropout": 0.0,
            "sculptor_s": 6,
            "sculptor_c": 2,
            "sculptor_a": 0,
        }

    @heavy_test
    def test_every_registry_id_resolves_managers_batches_and_forwards(self):
        import numpy as np
        import torch
        import yaml

        from D4CMPP2.src.utils import module_loader

        with open(ROOT / "network_refer.yaml", encoding="utf-8") as stream:
            registry = yaml.safe_load(stream)
        self.assertEqual(set(registry), EXPECTED_IDS)

        with tempfile.TemporaryDirectory(prefix="d4cmpp2-registry-smoke-") as temporary:
            model_path = Path(temporary)
            shutil.copyfile(
                ROOT / "src" / "utils" / "functional_group.csv",
                model_path / "functional_group.csv",
            )

            for network_id, entry in registry.items():
                with self.subTest(network=network_id):
                    config = self._base_config(model_path)
                    config.update(entry)
                    config["network_id"] = network_id

                    data_manager_type = module_loader.load_data_manager(config)
                    network_manager_type = module_loader.load_network_manager(config)
                    train_manager_type = module_loader.load_train_manager(config)
                    self.assertEqual(data_manager_type.__name__, entry["data_manager_class"])
                    self.assertEqual(network_manager_type.__name__, entry["network_manager_class"])
                    self.assertEqual(train_manager_type.__name__, entry["train_manager_class"])

                    manager = data_manager_type(config)
                    smiles = {
                        "compound": ["CCOCC", "CCNCC"]
                        if entry["data_manager_class"] == "ISADataManager"
                        else ["CCO", "CC"]
                    }
                    if "solvent" in manager.molecule_columns:
                        smiles["solvent"] = ["O", "CO"]

                    manager._molecule_smiles = {
                        column: np.asarray(values) for column, values in smiles.items()
                    }
                    manager.molecule_graphs = {}
                    for column in manager.molecule_columns:
                        generator = manager.gg_solv if column == "solvent" else manager.gg
                        manager.molecule_graphs[column] = [
                            generator.get_graph(value, **config) for value in smiles[column]
                        ]
                    manager.target_value = np.asarray([[0.25], [-0.5]], dtype=np.float32)
                    manager.original_row_indices = np.asarray([3, 7])

                    dataset = manager.init_dataset()
                    batch = manager.dataset.collate([dataset[0], dataset[1]])
                    if isinstance(batch, dict):
                        unwrapped = manager.unwrapper(device="cpu", **batch)
                    else:
                        unwrapped = manager.unwrapper(*batch, device="cpu")

                    network_module = importlib.import_module(
                        f"D4CMPP2.networks.{entry['network']}"
                    )
                    model = network_module.network(config)
                    output = model(**unwrapped)
                    self.assertEqual(tuple(output.shape), (2, 1))
                    self.assertTrue(torch.isfinite(output).all().item())
                    loss = model.loss_fn(output, unwrapped["target"])
                    self.assertTrue(torch.isfinite(loss).item())
                    loss.backward()
                    self.assertTrue(
                        all(
                            parameter.grad is None
                            or torch.isfinite(parameter.grad).all().item()
                            for parameter in model.parameters()
                        )
                    )

    @heavy_test
    def test_isatpn_rejects_multi_target_configuration(self):
        from D4CMPP2.networks.ISATPN_model import network

        config = {
            "target_dim": 2,
            "node_dim": 1,
            "edge_dim": 1,
            "hidden_dim": 8,
            "conv_layers": 1,
            "linear_layers": 1,
            "dropout": 0.0,
        }
        with self.assertRaisesRegex(
            ValueError,
            "requires target_dim=1.*positive and negative branches",
        ):
            network(config)


if __name__ == "__main__":
    unittest.main()
