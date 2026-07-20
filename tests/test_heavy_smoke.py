import importlib.util
import unittest

from markers import heavy_test


class HeavyDependencySmokeTests(unittest.TestCase):
    @heavy_test
    def test_rdkit_can_parse_tiny_fixture_smiles(self):
        if importlib.util.find_spec("rdkit") is None:
            self.fail("D4CMPP2_RUN_HEAVY=1 but RDKit is not installed")
        from rdkit import Chem

        self.assertIsNotNone(Chem.MolFromSmiles("CCO"))

    @heavy_test
    def test_pyg_backend_can_create_cpu_graph(self):
        if importlib.util.find_spec("torch") is None:
            self.fail("D4CMPP2_RUN_HEAVY=1 but PyTorch is not installed")
        if importlib.util.find_spec("torch_geometric") is None:
            self.fail("D4CMPP2_RUN_HEAVY=1 but PyTorch Geometric is not installed")
        import torch
        from torch_geometric.data import Data

        graph = Data(edge_index=torch.tensor([[0, 1], [1, 0]]), num_nodes=2)
        self.assertEqual(graph.edge_index.device.type, "cpu")
        self.assertEqual(graph.num_edges, 2)


if __name__ == "__main__":
    unittest.main()
