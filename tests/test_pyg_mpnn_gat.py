import sys
import unittest

from fixtures import ROOT
from markers import heavy_test


class PyGMPNNGATTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if str(ROOT.parent) not in sys.path:
            sys.path.insert(0, str(ROOT.parent))

    def _batch(self):
        from torch_geometric.data import Batch
        from D4CMPP2.src.DataManager.GraphGenerator.MolGraphGenerator import MolGraphGenerator

        generator = MolGraphGenerator()
        compound = Batch.from_data_list([generator.get_graph("CCO"), generator.get_graph("CC")])
        solvent = Batch.from_data_list([generator.get_graph("O"), generator.get_graph("CO")])
        return generator, compound, solvent

    @heavy_test
    def test_mpnn_layer_matches_destination_mean_definition(self):
        import torch
        import torch.nn as nn
        from torch_geometric.data import Data
        from D4CMPP2.networks.src.MPNN import MPNN_layer

        graph = Data(edge_index=torch.tensor([[0, 2, 1], [1, 1, 2]]), num_nodes=3)
        nodes = torch.tensor([[1.0], [2.0], [4.0]], requires_grad=True)
        edges = torch.tensor([[0.5], [1.5], [2.5]])
        layer = MPNN_layer(1, 1, 1, nn.Identity(), dropout=0.0)
        with torch.no_grad():
            layer.linear.weight.copy_(torch.tensor([[1.0, 2.0, 3.0]]))
            layer.linear.bias.copy_(torch.tensor([0.25]))
        messages = layer.linear(torch.cat([nodes[graph.edge_index[0]], nodes[graph.edge_index[1]], edges], dim=1))
        expected = torch.stack([torch.zeros_like(messages[0]), messages[:2].mean(0), messages[2]])
        observed = layer(graph, nodes, edges)
        self.assertTrue(torch.allclose(observed, expected, atol=1e-6, rtol=0.0))
        observed.sum().backward()
        self.assertTrue(torch.isfinite(nodes.grad).all().item())

    @heavy_test
    def test_gat_layer_matches_destination_softmax_definition(self):
        import torch
        import torch.nn as nn
        from torch_geometric.data import Data
        from D4CMPP2.networks.src.GAT import GAT_layer

        graph = Data(edge_index=torch.tensor([[0, 2, 1], [1, 1, 2]]), num_nodes=3)
        nodes = torch.tensor([[1.0], [2.0], [4.0]], requires_grad=True)
        layer = GAT_layer(1, 1, 1, nn.Identity(), dropout=0.0)
        with torch.no_grad():
            layer.attention_W.weight.copy_(torch.tensor([[1.0, 0.5]]))
            layer.attention_W.bias.zero_()
            layer.attention_a.weight.fill_(1.0)
            layer.linear.weight.fill_(2.0)
            layer.linear.bias.fill_(0.25)
        src, dst = graph.edge_index
        logits = layer.attention_a(nn.LeakyReLU()(layer.attention_W(torch.cat([nodes[src], nodes[dst]], dim=1))))
        weights = torch.cat([torch.softmax(logits[:2], dim=0), torch.ones_like(logits[2:])])
        expected = torch.stack([
            torch.zeros(1),
            (weights[:2] * layer.linear(nodes[src[:2]])).sum(0),
            layer.linear(nodes[src[2:]])[0],
        ])
        expected = nn.LeakyReLU()(expected)
        observed = layer(graph, nodes)
        self.assertTrue(torch.allclose(observed, expected, atol=1e-6, rtol=0.0))
        observed.sum().backward()
        self.assertTrue(torch.isfinite(nodes.grad).all().item())

    @heavy_test
    def test_network_contracts_and_parameter_counts(self):
        import torch
        from D4CMPP2.networks.MPNN_model import network as MPNN
        from D4CMPP2.networks.MPNNwithSolv_model import network as MPNNwS
        from D4CMPP2.networks.GAT_model import network as GAT
        from D4CMPP2.networks.GATwithSolv_model import network as GATwS

        generator, compound, solvent = self._batch()
        config = {"node_dim": generator.node_dim, "edge_dim": generator.edge_dim, "target_dim": 2,
                  "hidden_dim": 8, "conv_layers": 1, "linear_layers": 1, "dropout": 0.0}
        cases = [(MPNN, 658, True, False), (MPNNwS, 29130, True, True),
                 (GAT, 602, False, False), (GATwS, 29002, False, True)]
        for model_type, parameter_count, uses_edges, uses_solvent in cases:
            with self.subTest(model=model_type.__module__):
                model = model_type(config)
                self.assertEqual(sum(p.numel() for p in model.parameters()), parameter_count)
                kwargs = {"compound_graphs": compound, "compound_node_feature": compound.x}
                if uses_edges:
                    kwargs["compound_edge_feature"] = compound.edge_attr
                if uses_solvent:
                    kwargs.update(solvent_graphs=solvent, solvent_node_feature=solvent.x)
                output = model(**kwargs)
                self.assertEqual(tuple(output.shape), (2, 2))
                loss = model.loss_fn(output, torch.tensor([[0.25, float("nan")], [-0.5, 1.0]]))
                self.assertTrue(torch.isfinite(loss).item())
                loss.backward()
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
