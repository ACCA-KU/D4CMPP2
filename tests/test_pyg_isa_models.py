import sys
import unittest

from fixtures import ROOT
from markers import heavy_test


class PyGISAModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if str(ROOT.parent) not in sys.path:
            sys.path.insert(0, str(ROOT.parent))

    def _graph_fixture(self, smiles):
        import torch
        from torch_geometric.data import Batch
        from D4CMPP2.src.DataManager.GraphGenerator.ISAGraphGenerator import ISAGraphGenerator

        generator = ISAGraphGenerator(
            frag_ref=str(ROOT / "src" / "utils" / "functional_group.csv"), sculptor_index=(6, 2, 0)
        )
        graph = Batch.from_data_list([generator.get_graph(s, max_dist=4) for s in smiles])
        config = {"node_dim": generator.node_dim, "edge_dim": generator.edge_dim, "target_dim": 1,
                  "hidden_dim": 8, "conv_layers": 1, "linear_layers": 1, "dropout": 0.0}
        kwargs = {
            "compound_graphs": graph,
            "compound_r_node": graph["r_nd"].x,
            "compound_r2r_edge": graph["r_nd", "r2r", "r_nd"].edge_attr,
            "compound_i_node": graph["i_nd"].x,
            "compound_d_node": graph["d_nd"].x,
            "compound_d2d_edge": graph["d_nd", "d2d", "d_nd"].edge_attr,
        }
        return graph, config, kwargs

    def _fixture(self):
        import torch

        graph, config, kwargs = self._graph_fixture(("CCOCC", "CCNCC"))
        return graph, config, kwargs, torch.tensor([[0.25], [-0.5]])

    @heavy_test
    def test_relation_views_and_typed_sum_preserve_batch_contract(self):
        import torch
        from D4CMPP2.networks.src.pyg_hetero import relation_graph, relation_sum

        graph, _, _, _ = self._fixture()
        view = relation_graph(graph, "r_nd", "r2r")
        self.assertTrue(torch.equal(view.edge_index, graph["r_nd", "r2r", "r_nd"].edge_index))
        self.assertTrue(torch.equal(view.batch, graph["r_nd"].batch))
        source = torch.arange(graph["i_nd"].num_nodes, dtype=torch.float32).unsqueeze(1)
        observed = relation_sum(graph, "i_nd", "i2d", "d_nd", source)
        src, dst = graph["i_nd", "i2d", "d_nd"].edge_index
        expected = torch.zeros_like(observed)
        expected.index_add_(0, dst, source[src])
        self.assertTrue(torch.equal(observed, expected))

    @heavy_test
    def test_gc_isat_isatpn_contracts_and_parameter_counts(self):
        import torch
        from D4CMPP2.networks.GC_model import network as GC
        from D4CMPP2.networks.ISAT_model import network as ISAT
        from D4CMPP2.networks.ISATPN_model import network as ISATPN

        _, config, kwargs, target = self._fixture()
        for model_type, expected_count in ((GC, 869), (ISAT, 868), (ISATPN, 1155)):
            with self.subTest(model=model_type.__module__):
                torch.manual_seed(42)
                model = model_type(dict(config))
                model.train()
                self.assertEqual(sum(p.numel() for p in model.parameters()), expected_count)
                output = model(**kwargs)
                self.assertEqual(tuple(output.shape), (2, 1))
                self.assertTrue(torch.isfinite(output).all().item())
                loss = model.loss_fn(output, target)
                self.assertTrue(torch.isfinite(loss).item())
                loss.backward()
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()))

    @heavy_test
    def test_isatpn_compute_loss_preserves_all_regularization_terms(self):
        import torch
        from D4CMPP2.networks.ISATPN_model import ISATPN

        _, config, kwargs, target = self._fixture()
        model = ISATPN(config)
        model.eval()
        prediction = model(**kwargs)

        prediction_term = torch.mean((prediction - target) ** 2)
        magnitude_term = (
            model.p_score_ms + model.n_score_ms
        ) * model.alpha
        variance_term = (
            torch.abs(model.score_var - model.p_score_var)
            + torch.abs(model.score_var - model.n_score_var)
        ) * model.beta
        mean_term = (
            torch.abs(model.score_mean - model.p_score_mean)
            + torch.abs(model.score_mean - model.n_score_mean)
        ) * model.gamma
        expected = prediction_term + magnitude_term + variance_term + mean_term

        observed = model.compute_loss(prediction, target)
        self.assertTrue(torch.allclose(observed, expected, atol=1e-7, rtol=0.0))
        self.assertGreater(float(observed - prediction_term), 0.0)
        scores = model(**kwargs, get_score=True)
        self.assertTrue(
            torch.equal(
                model.ISATconv_PM.p_score_var,
                torch.var(scores["positive"].view(-1)),
            )
        )
        self.assertTrue(
            torch.equal(
                model.ISATconv_PM.n_score_var,
                torch.var(scores["negative"].view(-1)),
            )
        )
        with self.assertRaisesRegex(ValueError, "no finite target"):
            model.compute_loss(prediction, torch.full_like(target, float("nan")))

    @heavy_test
    def test_isatpn_single_fragment_variance_and_loss_are_finite(self):
        import torch
        from D4CMPP2.networks.ISATPN_model import ISATPN

        graph, config, kwargs = self._graph_fixture(("C",))
        self.assertEqual(graph["d_nd"].num_nodes, 1)
        model = ISATPN(config)
        model.eval()

        prediction = model(**kwargs)
        loss = model.compute_loss(prediction, torch.tensor([[0.25]]))

        self.assertEqual(float(model.p_score_var), 0.0)
        self.assertEqual(float(model.n_score_var), 0.0)
        self.assertTrue(torch.isfinite(loss).item())


if __name__ == "__main__":
    unittest.main()
