import sys
import tempfile
import unittest
from pathlib import Path

from fixtures import ROOT
from markers import heavy_test


class PyGDataContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if str(ROOT.parent) not in sys.path:
            sys.path.insert(0, str(ROOT.parent))

    @heavy_test
    def test_homogeneous_graph_preserves_dgl_topology_and_feature_order(self):
        import torch

        from D4CMPP2.src.DataManager.GraphGenerator.MolGraphGenerator import MolGraphGenerator

        generator = MolGraphGenerator()
        graph = generator.get_graph("CCO")

        self.assertEqual(graph.num_nodes, 3)
        self.assertEqual(graph.num_edges, 4)
        self.assertEqual(tuple(graph.x.shape), (3, generator.node_dim))
        self.assertEqual(tuple(graph.edge_attr.shape), (4, generator.edge_dim))
        self.assertTrue(
            torch.equal(
                graph.edge_index,
                torch.tensor([[0, 1, 1, 2], [1, 2, 0, 1]], dtype=torch.long),
            )
        )
        self.assertTrue(torch.equal(graph.edge_attr[:2], graph.edge_attr[2:]))

    @heavy_test
    def test_general_dataset_batches_graph_numeric_target_and_smiles(self):
        import torch

        from D4CMPP2.src.DataManager.Dataset.GraphDataset import GraphDataset
        from D4CMPP2.src.DataManager.GraphGenerator.MolGraphGenerator import MolGraphGenerator

        generator = MolGraphGenerator()
        graphs = {"compound": [generator.get_graph("CCO"), generator.get_graph("CC")]}
        dataset = GraphDataset(
            graphs=graphs,
            numeric_inputs={"temperature": torch.tensor([[298.0], [310.0]])},
            target=[[1.0, float("nan")], [2.0, 3.0]],
            smiles={"compound": ["CCO", "CC"]},
            row_indices=[7, 11],
        )
        batch = GraphDataset.collate([dataset[0], dataset[1]])
        unwrapped = GraphDataset.unwrapper(device="cpu", **batch)

        graph = unwrapped["compound_graphs"]
        self.assertEqual(graph.num_graphs, 2)
        self.assertEqual(graph.ptr.tolist(), [0, 3, 5])
        self.assertEqual(graph.batch.tolist(), [0, 0, 0, 1, 1])
        self.assertEqual(tuple(unwrapped["compound_node_feature"].shape), (5, generator.node_dim))
        self.assertEqual(tuple(unwrapped["compound_edge_feature"].shape), (6, generator.edge_dim))
        self.assertEqual(tuple(unwrapped["temperature_var"].shape), (2, 1))
        self.assertEqual(tuple(unwrapped["target"].shape), (2, 2))
        self.assertEqual(unwrapped["compound_smiles"], ["CCO", "CC"])
        self.assertEqual(unwrapped["original_row_index"].tolist(), [7, 11])

    @heavy_test
    def test_solvent_dataset_uses_two_aligned_pyg_batches(self):
        import importlib

        import torch

        from D4CMPP2.src.DataManager.Dataset.GraphDataset import GraphDataset_withSolv
        from D4CMPP2.src.DataManager.GraphGenerator.MolGraphGenerator import MolGraphGenerator
        from D4CMPP2.src.NetworkManager.NetworkManager import NetworkManager

        generator = MolGraphGenerator()
        compounds = [generator.get_graph("CCO"), generator.get_graph("CC")]
        solvents = [generator.get_graph("O"), generator.get_graph("CO")]
        dataset = GraphDataset_withSolv(
            graphs=compounds,
            solv_graphs=solvents,
            target=[[1.0], [2.0]],
            smiles=["CCO", "CC"],
            solv_smiles=["O", "CO"],
        )
        batch = GraphDataset_withSolv.collate([dataset[0], dataset[1]])
        unwrapped = GraphDataset_withSolv.unwrapper(*batch, device="cpu")

        self.assertEqual(unwrapped["graph"].num_graphs, 2)
        self.assertEqual(unwrapped["solv_graph"].num_graphs, 2)
        self.assertEqual(unwrapped["graph"].ptr.tolist(), [0, 3, 5])
        self.assertEqual(unwrapped["solv_graph"].ptr.tolist(), [0, 3, 5])
        self.assertEqual(unwrapped["smiles"], ["CCO", "CC"])
        self.assertEqual(unwrapped["solv_smiles"], ["O", "CO"])
        self.assertIs(unwrapped["compound_graphs"], unwrapped["graph"])
        self.assertIs(unwrapped["compound_node_feature"], unwrapped["node_feats"])
        self.assertIs(unwrapped["compound_edge_feature"], unwrapped["edge_feats"])
        self.assertIs(unwrapped["compound_smiles"], unwrapped["smiles"])
        self.assertIs(unwrapped["solvent_graphs"], unwrapped["solv_graph"])
        self.assertIs(unwrapped["solvent_node_feature"], unwrapped["solv_node_feats"])
        self.assertIs(unwrapped["solvent_edge_feature"], unwrapped["solv_edge_feats"])
        self.assertIs(unwrapped["solvent_smiles"], unwrapped["solv_smiles"])

        config = {
            "node_dim": generator.node_dim,
            "edge_dim": generator.edge_dim,
            "target_dim": 1,
            "hidden_dim": 8,
            "conv_layers": 1,
            "linear_layers": 1,
            "dropout": 0.0,
        }
        for module_name in (
            "GCNwithSolv_model",
            "MPNNwithSolv_model",
            "DMPNNwithSolv_model",
            "AFPwithSolv_model",
            "GATwithSolv_model",
        ):
            with self.subTest(network=module_name):
                model = importlib.import_module(
                    f"D4CMPP2.networks.{module_name}"
                ).network(config)
                network_manager = NetworkManager.__new__(NetworkManager)
                network_manager.network = model
                network_manager.device = "cpu"
                network_manager.unwrapper = GraphDataset_withSolv.unwrapper
                network_manager.optimizer = torch.optim.Adam(model.parameters())
                network_manager.loss_fn = model.loss_fn
                network_manager.state = "eval"
                target, output, loss = network_manager.step(batch)
                self.assertEqual(tuple(target.shape), (2, 1))
                self.assertEqual(tuple(output.shape), (2, 1))
                self.assertTrue(torch.isfinite(torch.tensor(loss)).item())

    @heavy_test
    def test_empty_graph_has_explicit_feature_shapes(self):
        from D4CMPP2.src.DataManager.GraphGenerator.MolGraphGenerator import MolGraphGenerator

        generator = MolGraphGenerator()
        graph = generator.get_empty_graph()
        self.assertEqual(graph.num_nodes, 0)
        self.assertEqual(tuple(graph.edge_index.shape), (2, 0))
        self.assertEqual(tuple(graph.x.shape), (0, generator.node_dim))
        self.assertEqual(tuple(graph.edge_attr.shape), (0, generator.edge_dim))

    @heavy_test
    def test_isa_heterodata_preserves_types_relations_and_batch_membership(self):
        import torch
        from torch_geometric.data import Batch

        from D4CMPP2.src.DataManager.GraphGenerator.ISAGraphGenerator import ISAGraphGenerator

        generator = ISAGraphGenerator(
            frag_ref=str(ROOT / "src" / "utils" / "functional_group.csv"),
            sculptor_index=(6, 2, 0),
        )
        graphs = [generator.get_graph(smiles, max_dist=4) for smiles in ("CCOCC", "CCNCC")]
        graph = graphs[0]

        self.assertEqual(set(graph.node_types), {"r_nd", "i_nd", "d_nd"})
        self.assertEqual(
            set(graph.edge_types),
            {
                ("r_nd", "r2r", "r_nd"),
                ("r_nd", "r2i", "i_nd"),
                ("i_nd", "i2i", "i_nd"),
                ("i_nd", "i2d", "d_nd"),
                ("d_nd", "d2d", "d_nd"),
                ("d_nd", "d2r", "r_nd"),
            },
        )
        self.assertEqual(graph["r_nd"].num_nodes, 5)
        self.assertEqual(graph["i_nd"].num_nodes, 5)
        self.assertEqual(tuple(graph["r_nd"].x.shape), (5, generator.r_node_dim))
        self.assertEqual(
            graph["r_nd", "r2i", "i_nd"].edge_index.tolist(),
            [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],
        )
        distances = graph["d_nd", "d2d", "d_nd"].edge_attr
        self.assertEqual(distances.shape[1], 4)
        self.assertTrue(torch.all((distances == 0) | (distances == 1)).item())
        self.assertTrue(torch.all(distances.sum(dim=1) <= 1).item())

        batch = Batch.from_data_list(graphs)
        self.assertEqual(batch.num_graphs, 2)
        self.assertEqual(batch["r_nd"].ptr.tolist(), [0, 5, 10])
        self.assertEqual(batch["r_nd"].batch.tolist(), [0] * 5 + [1] * 5)

    @heavy_test
    def test_isa_single_atom_self_loop_has_aligned_edge_features(self):
        import torch

        from D4CMPP2.src.DataManager.GraphGenerator.ISAGraphGenerator import ISAGraphGenerator

        generator = ISAGraphGenerator(
            frag_ref=str(ROOT / "src" / "utils" / "functional_group.csv"),
            sculptor_index=(6, 2, 0),
        )
        graph = generator.get_graph("C", max_dist=4)
        relation = graph["r_nd", "r2r", "r_nd"]

        self.assertEqual(relation.edge_index.tolist(), [[0], [0]])
        self.assertEqual(tuple(relation.edge_attr.shape), (1, generator.edge_dim))
        self.assertTrue(torch.equal(relation.edge_attr, torch.zeros_like(relation.edge_attr)))

    @heavy_test
    def test_isa_dataset_batches_features_target_and_smiles(self):
        from D4CMPP2.src.DataManager.Dataset.ISAGraphDataset import ISAGraphDataset
        from D4CMPP2.src.DataManager.GraphGenerator.ISAGraphGenerator import ISAGraphGenerator

        generator = ISAGraphGenerator(
            frag_ref=str(ROOT / "src" / "utils" / "functional_group.csv"),
            sculptor_index=(6, 2, 0),
        )
        graphs = {"compound": [generator.get_graph("CCOCC"), generator.get_graph("CCNCC")]}
        dataset = ISAGraphDataset(
            graphs=graphs,
            target=[[1.0], [2.0]],
            smiles={"compound": ["CCOCC", "CCNCC"]},
            row_indices=[3, 9],
        )
        batch = ISAGraphDataset.collate([dataset[0], dataset[1]])
        unwrapped = ISAGraphDataset.unwrapper(device="cpu", **batch)

        self.assertEqual(unwrapped["compound_graphs"].num_graphs, 2)
        self.assertEqual(tuple(unwrapped["compound_r_node"].shape), (10, generator.r_node_dim))
        self.assertEqual(tuple(unwrapped["target"].shape), (2, 1))
        self.assertEqual(unwrapped["compound_smiles"], ["CCOCC", "CCNCC"])
        self.assertEqual(unwrapped["original_row_index"].tolist(), [3, 9])

    @heavy_test
    def test_pyg_cache_round_trip_and_schema_rejection(self):
        import torch

        from D4CMPP2.src.DataManager.GraphGenerator.MolGraphGenerator import MolGraphGenerator
        from D4CMPP2.src.DataManager.MolDataManager import MolDataManager

        generator = MolGraphGenerator()
        with tempfile.TemporaryDirectory(prefix="d4cmpp2-pyg-cache-") as temporary:
            manager = object.__new__(MolDataManager)
            manager.data = "fixture"
            manager.graph_type = "mol"
            manager.explicit_h_columns = []
            manager.config = {"GRAPH_DIR": temporary}
            manager.gg = generator
            manager._molecule_smiles = {"compound": ["CCO", "CC"]}
            manager.molecule_graphs = {
                "compound": [generator.get_graph("CCO"), generator.get_graph("CC")]
            }
            manager.graph_errors = []
            manager.original_row_indices = torch.tensor([4, 8])

            manager.save_graphs("compound")
            cache_path = Path(manager.get_graphs_path("compound"))
            self.assertRegex(
                cache_path.name,
                r"^fixture_compound_mol_pyg_v2_[0-9a-f]{16}\.pt$",
            )
            self.assertTrue(cache_path.is_file())

            manager.molecule_graphs = {"compound": []}
            manager.load_graphs("compound")
            self.assertEqual([graph.num_nodes for graph in manager.molecule_graphs["compound"]], [3, 2])
            self.assertEqual(manager.molecule_graphs["compound"][0].edge_index.tolist(), [[0, 1, 1, 2], [1, 2, 0, 1]])

            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            payload["recipe"]["feature_contract_version"] = 99
            torch.save(payload, cache_path)
            with self.assertRaisesRegex(ValueError, "feature_contract_version.*Regenerate"):
                manager.load_graphs("compound")

    @heavy_test
    def test_pyg_batches_move_to_available_device(self):
        import torch

        from D4CMPP2.src.DataManager.Dataset.GraphDataset import GraphDataset
        from D4CMPP2.src.DataManager.GraphGenerator.MolGraphGenerator import MolGraphGenerator

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        generator = MolGraphGenerator()
        dataset = GraphDataset(
            graphs={"compound": [generator.get_graph("CCO"), generator.get_graph("CC")]},
            target=[[1.0], [2.0]],
            smiles={"compound": ["CCO", "CC"]},
            row_indices=[1, 2],
        )
        batch = GraphDataset.collate([dataset[0], dataset[1]])
        unwrapped = GraphDataset.unwrapper(device=device, **batch)
        self.assertEqual(unwrapped["compound_graphs"].x.device.type, device.split(":")[0])
        self.assertEqual(unwrapped["compound_node_feature"].device.type, device.split(":")[0])
        self.assertEqual(unwrapped["target"].device.type, device.split(":")[0])
        self.assertEqual(unwrapped["original_row_index"].device.type, device.split(":")[0])


if __name__ == "__main__":
    unittest.main()
