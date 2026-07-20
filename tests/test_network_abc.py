import unittest

from markers import heavy_test


class NetworkAbcTests(unittest.TestCase):
    @heavy_test
    def test_gcn_contract_config_and_search_space_are_explicit(self):
        import torch

        from D4CMPP2.networks.GCN_model import GCN
        from D4CMPP2.networks.base import MolecularNetwork
        from D4CMPP2.networks.registry import get_model

        source = {"node_dim": 10, "target_dim": 2, "hidden_dim": 32}
        model = GCN(source)
        self.assertIsInstance(model, MolecularNetwork)
        self.assertEqual(model.config["hidden_dim"], 32)
        self.assertEqual(model.config["conv_layers"], 4)
        self.assertNotIn("conv_layers", source)
        self.assertIs(get_model("GCN").network, GCN)
        self.assertIs(get_model("gcn").network, GCN)
        self.assertEqual(
            set(GCN.optimization_space()),
            {"hidden_dim", "conv_layers", "linear_layers", "dropout"},
        )
        with self.assertRaisesRegex(ValueError, "Available keys"):
            GCN.optimization_space(["unknown"])
        with self.assertRaisesRegex(ValueError, "missing required fields"):
            GCN({"target_dim": 1})
        with self.assertRaisesRegex(ValueError, "missing required fields"):
            model(compound_node_feature=torch.ones(2, 10))

    @heavy_test
    def test_registry_rejects_non_abc_and_name_mismatch(self):
        import torch.nn as nn

        from D4CMPP2.networks.GCN_model import GCN
        from D4CMPP2.networks.registry import ModelDefinition

        with self.assertRaisesRegex(TypeError, "inherit MolecularNetwork"):
            ModelDefinition("plain", nn.Linear)
        with self.assertRaisesRegex(ValueError, "does not match"):
            ModelDefinition("not-gcn", GCN)


if __name__ == "__main__":
    unittest.main()
