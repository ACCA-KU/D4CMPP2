import unittest

from fixtures import ROOT
from markers import heavy_test


class CustomNetworkTests(unittest.TestCase):
    @heavy_test
    def test_custom_network_registers_builds_and_uses_its_own_loss(self):
        import torch
        from torch_geometric.data import Batch, Data

        import D4CMPP2
        from D4CMPP2.examples.custom_network import CustomGCN
        from D4CMPP2.networks.registry import get_model

        D4CMPP2.register_network(
            CustomGCN,
            aliases=("CustomGCN",),
            data_contract="molecule",
        )
        definition = get_model("CustomGCN")
        self.assertIs(definition.network, CustomGCN)
        self.assertEqual(definition.data_contract, "molecule")
        self.assertEqual(
            definition.training_config()["data_manager_class"],
            "MolDataManager",
        )

        source = {"node_dim": 3, "target_dim": 1, "huber_delta": 0.5}
        model = definition.network(source)
        self.assertNotIn("hidden_dim", source)
        self.assertEqual(
            set(model.optimization_space()),
            {"hidden_dim", "dropout", "huber_delta"},
        )

        graphs = Batch.from_data_list(
            [
                Data(x=torch.ones(2, 3), edge_index=torch.empty((2, 0), dtype=torch.long)),
                Data(x=torch.full((1, 3), 2.0), edge_index=torch.empty((2, 0), dtype=torch.long)),
            ]
        )
        prediction = model(
            compound_graphs=graphs,
            compound_node_feature=graphs.x,
        )
        target = torch.tensor([[10.0], [float("nan")]])
        observed = model.compute_loss(prediction, target)
        mse = torch.mean((prediction[0] - target[0]) ** 2)
        self.assertTrue(torch.isfinite(observed).item())
        self.assertFalse(torch.allclose(observed, mse))
        observed.backward()

        with self.assertRaisesRegex(ValueError, "already registered"):
            D4CMPP2.register_network(CustomGCN)

    @heavy_test
    def test_registered_custom_network_supplies_training_manager_contract(self):
        import D4CMPP2
        from D4CMPP2._main import load_NET_REFER
        from D4CMPP2.examples.custom_network import CustomGCN
        from D4CMPP2.networks.registry import get_model

        try:
            get_model("custom_gcn")
        except ValueError:
            D4CMPP2.register_network(CustomGCN, data_contract="molecule")

        config = load_NET_REFER(
            {
                "network": "custom_gcn",
                "NET_REFER": str(ROOT / "network_refer.yaml"),
            }
        )
        self.assertEqual(config["network_id"], "custom_gcn")
        self.assertEqual(config["data_manager_class"], "MolDataManager")
        self.assertEqual(config["network_manager_class"], "NetworkManager")
        self.assertEqual(config["train_manager_class"], "Trainer")


if __name__ == "__main__":
    unittest.main()
