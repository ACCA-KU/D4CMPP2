import sys
import unittest

from fixtures import ROOT
from markers import heavy_test


class PyGGcnTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if str(ROOT.parent) not in sys.path:
            sys.path.insert(0, str(ROOT.parent))

    @heavy_test
    def test_gcn_layer_matches_dgl_neighbor_sum_definition(self):
        import torch
        import torch.nn as nn

        from D4CMPP2.networks.src.GCN import GCN_layer
        from D4CMPP2.src.DataManager.GraphGenerator.MolGraphGenerator import MolGraphGenerator

        graph = MolGraphGenerator().get_graph("CCO")
        node_feats = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
        layer = GCN_layer(2, 2, nn.Identity(), dropout=0.0, batch_norm=False, residual_sum=False)
        with torch.no_grad():
            layer.linear.weight.copy_(torch.tensor([[1.0, 0.5], [-0.25, 1.0]]))
            layer.linear.bias.copy_(torch.tensor([0.1, -0.2]))

        transformed = layer.linear(node_feats)
        expected = torch.stack(
            [transformed[0] + transformed[1], transformed[0] + transformed[1] + transformed[2], transformed[1] + transformed[2]]
        )
        observed = layer(graph, node_feats)
        self.assertTrue(
            torch.allclose(observed, expected, atol=1e-6, rtol=0.0),
            f"max_abs_diff={torch.max(torch.abs(observed - expected)).item()}",
        )
        observed.sum().backward()
        self.assertTrue(torch.isfinite(node_feats.grad).all().item())

    def _batch(self):
        from torch_geometric.data import Batch

        from D4CMPP2.src.DataManager.GraphGenerator.MolGraphGenerator import MolGraphGenerator

        generator = MolGraphGenerator()
        compound = Batch.from_data_list([generator.get_graph("CCO"), generator.get_graph("CC")])
        solvent = Batch.from_data_list([generator.get_graph("O"), generator.get_graph("CO")])
        return generator, compound, solvent

    @heavy_test
    def test_gcn_network_preserves_parameter_shape_loss_and_backward_contract(self):
        import torch

        from D4CMPP2.networks.GCN_model import network

        torch.manual_seed(42)
        generator, compound, _ = self._batch()
        config = {
            "node_dim": generator.node_dim,
            "edge_dim": generator.edge_dim,
            "target_dim": 2,
            "hidden_dim": 8,
            "conv_layers": 1,
            "linear_layers": 1,
            "dropout": 0.0,
        }
        model = network(config)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 458)
        output = model(compound_graphs=compound, compound_node_feature=compound.x)
        self.assertEqual(tuple(output.shape), (2, 2))
        target = torch.tensor([[0.25, float("nan")], [-0.5, 1.0]])
        loss = model.loss_fn(output, target)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()))

    @heavy_test
    def test_solvent_gcn_preserves_two_stream_contract_and_parameter_count(self):
        import torch

        from D4CMPP2.networks.GCNwithSolv_model import network

        torch.manual_seed(42)
        generator, compound, solvent = self._batch()
        config = {
            "node_dim": generator.node_dim,
            "edge_dim": generator.edge_dim,
            "target_dim": 2,
            "hidden_dim": 8,
            "conv_layers": 1,
            "linear_layers": 1,
            "dropout": 0.0,
        }
        model = network(config)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 29690)
        output = model(
            compound_graphs=compound,
            compound_node_feature=compound.x,
            solvent_graphs=solvent,
            solvent_node_feature=solvent.x,
        )
        self.assertEqual(tuple(output.shape), (2, 2))
        loss = model.loss_fn(output, torch.tensor([[0.25, 0.75], [-0.5, 1.0]]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())


if __name__ == "__main__":
    unittest.main()
