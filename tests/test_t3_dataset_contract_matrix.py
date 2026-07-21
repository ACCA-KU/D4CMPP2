import sys
import unittest
import warnings

from fixtures import ROOT
from markers import heavy_test


class DatasetContractMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if str(ROOT.parent) not in sys.path:
            sys.path.insert(0, str(ROOT.parent))

    @heavy_test
    def test_tensor_targets_are_copied_without_pytorch_copy_warnings(self):
        import torch

        from D4CMPP2.src.DataManager.Dataset.GraphDataset import (
            GraphDataset,
            GraphDataset_legacy,
        )
        from D4CMPP2.src.DataManager.Dataset.ISAGraphDataset import ISAGraphDataset

        target = torch.tensor([[1.0], [2.0]], requires_grad=True)
        graph = type(
            "MinimalGraph",
            (),
            {"x": torch.ones((1, 1)), "edge_attr": torch.ones((1, 1))},
        )()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            legacy = GraphDataset_legacy([graph, graph], target, ["C", "CC"])
            general = GraphDataset({}, target=target, smiles={})
            isa = ISAGraphDataset({}, target=target, smiles={})

        copy_warnings = [
            warning
            for warning in caught
            if "copy construct from a tensor" in str(warning.message)
        ]
        self.assertEqual(copy_warnings, [])
        for copied in (legacy.target, general.target, isa.target):
            self.assertEqual(copied.dtype, torch.float32)
            self.assertFalse(copied.requires_grad)
            self.assertNotEqual(copied.data_ptr(), target.data_ptr())

    @heavy_test
    def test_generalized_dataset_preserves_multiple_molecules_numeric_and_subset_keys(self):
        import torch

        from D4CMPP2.src.DataManager.Dataset.GraphDataset import GraphDataset
        from D4CMPP2.src.DataManager.GraphGenerator.MolGraphGenerator import MolGraphGenerator

        generator = MolGraphGenerator()
        dataset = GraphDataset(
            graphs={
                "compound": [generator.get_graph("CC"), generator.get_graph("CCC")],
                "coformer": [generator.get_graph("O"), generator.get_graph("N")],
            },
            numeric_inputs={"temperature": [[298.0], [310.0]]},
            target=[[1.0, float("nan")], [2.0, 3.0]],
            smiles={"compound": ["CC", "CCC"], "coformer": ["O", "N"]},
            row_indices=[12, 19],
        )

        subset = dataset.get_subDataset([1])
        batch = GraphDataset.collate([subset[0]])
        unwrapped = GraphDataset.unwrapper(device="cpu", **batch)

        expected = {
            "compound_graphs",
            "compound_node_feature",
            "compound_edge_feature",
            "compound_smiles",
            "coformer_graphs",
            "coformer_node_feature",
            "coformer_edge_feature",
            "coformer_smiles",
            "temperature_var",
            "target",
            "original_row_index",
        }
        self.assertEqual(set(unwrapped), expected)
        self.assertEqual(unwrapped["compound_graphs"].num_graphs, 1)
        self.assertEqual(unwrapped["coformer_graphs"].num_graphs, 1)
        self.assertEqual(unwrapped["compound_smiles"], ["CCC"])
        self.assertEqual(unwrapped["coformer_smiles"], ["N"])
        self.assertEqual(tuple(unwrapped["temperature_var"].shape), (1, 1))
        self.assertEqual(unwrapped["temperature_var"].dtype, torch.float32)
        self.assertEqual(tuple(unwrapped["target"].shape), (1, 2))
        self.assertEqual(unwrapped["original_row_index"].dtype, torch.int64)
        self.assertEqual(unwrapped["original_row_index"].tolist(), [19])

    @heavy_test
    def test_legacy_general_and_solvent_unwrappers_keep_compatibility_keys(self):
        import torch

        from D4CMPP2.src.DataManager.Dataset.GraphDataset import (
            GraphDataset_legacy,
            GraphDataset_withSolv,
        )
        from D4CMPP2.src.DataManager.GraphGenerator.MolGraphGenerator import MolGraphGenerator

        generator = MolGraphGenerator()
        compounds = [generator.get_graph("CC"), generator.get_graph("CCC")]
        solvents = [generator.get_graph("O"), generator.get_graph("N")]

        general = GraphDataset_legacy(compounds, [[1.0], [2.0]], ["CC", "CCC"])
        general_batch = GraphDataset_legacy.collate([general[0], general[1]])
        general_data = GraphDataset_legacy.unwrapper(*general_batch, device="cpu")
        self.assertEqual(
            set(general_data),
            {"graph", "node_feats", "edge_feats", "target", "smiles"},
        )
        self.assertEqual(general_data["node_feats"].dtype, torch.float32)
        self.assertEqual(general_data["edge_feats"].dtype, torch.float32)
        self.assertEqual(tuple(general_data["target"].shape), (2, 1))

        solvent = GraphDataset_withSolv(
            compounds,
            solvents,
            [[1.0], [2.0]],
            ["CC", "CCC"],
            ["O", "N"],
        )
        solvent_batch = GraphDataset_withSolv.collate([solvent[0], solvent[1]])
        solvent_data = GraphDataset_withSolv.unwrapper(*solvent_batch, device="cpu")
        self.assertEqual(
            set(solvent_data),
            {
                "graph",
                "node_feats",
                "edge_feats",
                "solv_graph",
                "solv_node_feats",
                "solv_edge_feats",
                "target",
                "smiles",
                "solv_smiles",
                "compound_graphs",
                "compound_node_feature",
                "compound_edge_feature",
                "compound_smiles",
                "solvent_graphs",
                "solvent_node_feature",
                "solvent_edge_feature",
                "solvent_smiles",
            },
        )
        self.assertEqual(solvent_data["graph"].num_graphs, 2)
        self.assertEqual(solvent_data["solv_graph"].num_graphs, 2)
        self.assertIs(solvent_data["compound_graphs"], solvent_data["graph"])
        self.assertIs(solvent_data["solvent_graphs"], solvent_data["solv_graph"])

    @heavy_test
    def test_isa_generalized_dataset_preserves_numeric_subset_and_feature_contract(self):
        import torch

        from D4CMPP2.src.DataManager.Dataset.ISAGraphDataset import ISAGraphDataset
        from D4CMPP2.src.DataManager.GraphGenerator.ISAGraphGenerator import ISAGraphGenerator

        generator = ISAGraphGenerator(
            frag_ref=str(ROOT / "src" / "utils" / "functional_group.csv"),
            sculptor_index=(6, 2, 0),
        )
        dataset = ISAGraphDataset(
            graphs={
                "compound": [
                    generator.get_graph("CCOCC", max_dist=4),
                    generator.get_graph("CCNCC", max_dist=4),
                ]
            },
            numeric_inputs={"temperature": [[298.0], [310.0]]},
            target=[[1.0], [2.0]],
            smiles={"compound": ["CCOCC", "CCNCC"]},
            row_indices=[4, 8],
        )

        subset = dataset.get_subDataset([0])
        batch = ISAGraphDataset.collate([subset[0]])
        unwrapped = ISAGraphDataset.unwrapper(device="cpu", **batch)

        required = {
            "compound_graphs",
            "compound_r_node",
            "compound_r2r_edge",
            "compound_i_node",
            "compound_i2i_edge",
            "compound_d_node",
            "compound_d2d_edge",
            "compound_smiles",
            "temperature_var",
            "target",
            "original_row_index",
        }
        self.assertEqual(set(unwrapped), required)
        self.assertEqual(unwrapped["compound_graphs"].num_graphs, 1)
        self.assertEqual(unwrapped["temperature_var"].dtype, torch.float32)
        self.assertEqual(tuple(unwrapped["temperature_var"].shape), (1, 1))
        self.assertEqual(unwrapped["target"].dtype, torch.float32)
        self.assertEqual(unwrapped["original_row_index"].dtype, torch.int64)
        self.assertEqual(unwrapped["compound_smiles"], ["CCOCC"])

    @heavy_test
    def test_isa_legacy_unwrapper_keeps_compatibility_keys(self):
        import torch

        from D4CMPP2.src.DataManager.Dataset.ISAGraphDataset import ISAGraphDataset_legacy
        from D4CMPP2.src.DataManager.GraphGenerator.ISAGraphGenerator import ISAGraphGenerator

        generator = ISAGraphGenerator(
            frag_ref=str(ROOT / "src" / "utils" / "functional_group.csv"),
            sculptor_index=(6, 2, 0),
        )
        graphs = [
            generator.get_graph("CCOCC", max_dist=4),
            generator.get_graph("CCNCC", max_dist=4),
        ]
        dataset = ISAGraphDataset_legacy(
            graphs=graphs,
            target=[[1.0], [2.0]],
            smiles=["CCOCC", "CCNCC"],
        )
        batch = ISAGraphDataset_legacy.collate([dataset[0], dataset[1]])
        unwrapped = ISAGraphDataset_legacy.unwrapper(*batch, device="cpu")

        self.assertEqual(
            set(unwrapped),
            {"graph", "r_node", "r_edge", "i_node", "d_node", "d_edge", "target", "smiles"},
        )
        self.assertEqual(unwrapped["graph"].num_graphs, 2)
        for key in ("r_node", "r_edge", "i_node", "d_node", "d_edge", "target"):
            self.assertEqual(unwrapped[key].dtype, torch.float32)
        self.assertEqual(tuple(unwrapped["target"].shape), (2, 1))
        self.assertEqual(unwrapped["smiles"], ["CCOCC", "CCNCC"])

    @heavy_test
    def test_single_atom_explicit_hydrogen_and_empty_edge_shapes_are_stable(self):
        from D4CMPP2.src.DataManager.GraphGenerator.MolGraphGenerator import MolGraphGenerator

        generator = MolGraphGenerator()

        single_atom = generator.get_graph("O")
        self.assertEqual(single_atom.num_nodes, 3)
        self.assertEqual(tuple(single_atom.edge_index.shape), (2, 4))
        self.assertEqual(tuple(single_atom.edge_attr.shape), (4, generator.edge_dim))

        implicit = generator.get_graph("CC")
        explicit = generator.get_graph("CC", explicit_h=True)
        self.assertEqual(implicit.num_nodes, 2)
        self.assertEqual(explicit.num_nodes, 8)

        disconnected = generator.get_graph("C.C")
        self.assertEqual(disconnected.num_nodes, 2)
        self.assertEqual(tuple(disconnected.edge_index.shape), (2, 0))
        self.assertEqual(tuple(disconnected.edge_attr.shape), (0, generator.edge_dim))


if __name__ == "__main__":
    unittest.main()
