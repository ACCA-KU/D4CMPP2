import sys
import unittest

from fixtures import ROOT
from markers import heavy_test


class PyGDMPNNAFPTests(unittest.TestCase):
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
    def test_dmpnn_layer_matches_directed_edge_definition(self):
        import torch
        import torch.nn as nn
        from torch_geometric.data import Data
        from D4CMPP2.networks.src.DMPNN import DMPNNLayer

        graph = Data(edge_index=torch.tensor([[0, 1], [1, 0]]), num_nodes=2)
        node = torch.tensor([[1.0], [3.0]], requires_grad=True)
        edge = torch.tensor([[0.5], [1.5]])
        layer = DMPNNLayer(1, 1, 1, nn.Identity(), dropout=0.0)
        with torch.no_grad():
            layer.W_m.weight.copy_(torch.tensor([[2.0, 4.0]]))
            layer.W_m.bias.fill_(0.25)
        src, dst = graph.edge_index
        direct = layer.W_m(torch.cat([edge, node[src]], 1))
        backward = layer.W_m(torch.cat([edge, node[dst]], 1))
        full = torch.stack([direct[1], direct[0]])
        observed_node, observed_direct, observed_backward = layer(graph, node, edge)
        self.assertTrue(torch.allclose(observed_node, full))
        self.assertTrue(torch.allclose(observed_direct, full[src] - backward))
        self.assertTrue(torch.allclose(observed_backward, full[dst] - direct))
        observed_node.sum().backward()
        self.assertTrue(torch.isfinite(node.grad).all().item())

    @heavy_test
    def test_network_contracts_parameter_counts_and_afp_attention(self):
        import torch
        from D4CMPP2.networks.DMPNN_model import network as DMPNN
        from D4CMPP2.networks.DMPNNwithSolv_model import network as DMPNNwS
        from D4CMPP2.networks.AFP_model import network as AFP
        from D4CMPP2.networks.AFPwithSolv_model import network as AFPwS

        generator, compound, solvent = self._batch()
        config = {"node_dim": generator.node_dim, "edge_dim": generator.edge_dim, "target_dim": 2,
                  "hidden_dim": 8, "conv_layers": 1, "linear_layers": 1, "dropout": 0.0}
        cases = [(DMPNN, 866, False), (DMPNNwS, 29266, True),
                 (AFP, 1636, False), (AFPwS, 30036, True)]
        for model_type, parameter_count, uses_solvent in cases:
            with self.subTest(model=model_type.__module__):
                model = model_type(config)
                self.assertEqual(sum(p.numel() for p in model.parameters()), parameter_count)
                kwargs = {"compound_graphs": compound, "compound_node_feature": compound.x,
                          "compound_edge_feature": compound.edge_attr}
                if uses_solvent:
                    kwargs.update(solvent_graphs=solvent, solvent_node_feature=solvent.x)
                output = model(**kwargs)
                self.assertEqual(tuple(output.shape), (2, 2))
                loss = model.loss_fn(output, torch.tensor([[0.25, float("nan")], [-0.5, 1.0]]))
                loss.backward()
                self.assertTrue(torch.isfinite(loss).item())
        afp = AFP(config)
        scored = afp(compound_graphs=compound, compound_node_feature=compound.x,
                     compound_edge_feature=compound.edge_attr, get_score=True)
        weights = scored["positive"]
        sums = weights.new_zeros(compound.num_graphs)
        sums.index_add_(0, compound.batch, weights[:, 0])
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-6, rtol=0.0))


if __name__ == "__main__":
    unittest.main()
