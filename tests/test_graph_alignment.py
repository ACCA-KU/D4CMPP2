import unittest
import warnings

try:
    from .markers import heavy_test
except ImportError:
    from markers import heavy_test


class _Graph:
    def __init__(self, nodes):
        self.nodes = nodes

    def number_of_nodes(self):
        return self.nodes

    @property
    def num_nodes(self):
        return self.nodes


class _GraphGenerator:
    def get_graph(self, smiles, **kwargs):
        if smiles.startswith("invalid"):
            raise ValueError(f"cannot parse {smiles}")
        return _Graph(1)

    def get_empty_graph(self):
        return _Graph(0)


class MultiInputGraphAlignmentTests(unittest.TestCase):
    @heavy_test
    def test_one_combined_mask_keeps_graph_smiles_numeric_target_and_set_aligned(self):
        import numpy as np
        import pandas as pd
        import torch
        from D4CMPP2.src.DataManager.MolDataManager import MolDataManager

        manager = object.__new__(MolDataManager)
        manager.molecule_columns = ["compound", "solvent"]
        manager._molecule_smiles = {
            "compound": np.array(["c0", "c1", "c2", "c3"]),
            "solvent": np.array(["s0", "s1", "s2", "s3"]),
        }
        manager.molecule_smiles = {key: values.copy() for key, values in manager._molecule_smiles.items()}
        manager.valid_smiles = {key: values.copy() for key, values in manager._molecule_smiles.items()}
        manager.molecule_graphs = {
            "compound": [_Graph(1), _Graph(0), _Graph(1), _Graph(1)],
            "solvent": [_Graph(1), _Graph(1), _Graph(0), _Graph(1)],
        }
        manager.numeric_inputs = {"temperature": np.array([10, 11, 12, 13])}
        manager.target_value = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
        manager.set = pd.Series(["train", "train", "val", "test"])
        manager.original_row_indices = np.array([10, 11, 12, 13])

        manager.drop_none_graph()

        self.assertEqual(manager._molecule_smiles["compound"].tolist(), ["c0", "c3"])
        self.assertEqual(manager._molecule_smiles["solvent"].tolist(), ["s0", "s3"])
        self.assertEqual(manager.numeric_inputs["temperature"].tolist(), [10, 13])
        self.assertEqual(manager.target_value[:, 0].tolist(), [0.0, 3.0])
        self.assertEqual(manager.set.tolist(), ["train", "test"])
        self.assertEqual(manager.original_row_indices.tolist(), [10, 13])
        self.assertEqual([graph.nodes for graph in manager.molecule_graphs["compound"]], [1, 1])
        self.assertEqual([graph.nodes for graph in manager.molecule_graphs["solvent"]], [1, 1])

    @heavy_test
    def test_multi_column_errors_are_aggregated_under_model_path_with_row_indices(self):
        import contextlib
        import io
        import numpy as np
        import pandas as pd
        from pathlib import Path
        import tempfile
        from D4CMPP2.src.DataManager.MolDataManager import MolDataManager

        with tempfile.TemporaryDirectory(prefix="d4cmpp2-graph-errors-") as temporary:
            root = Path(temporary)
            model_path = root / "model"
            model_path.mkdir()
            manager = object.__new__(MolDataManager)
            manager.config = {"MODEL_PATH": str(model_path)}
            manager.explicit_h_columns = []
            manager.gg = _GraphGenerator()
            manager.gg_solv = _GraphGenerator()
            manager.graph_errors = []
            manager.original_row_indices = np.array([100, 101, 102])
            manager._molecule_smiles = {
                "compound": np.array(["CC", "invalid-compound", "CCC"]),
                "solvent": np.array(["invalid-solvent", "O", "N"]),
            }
            manager.molecule_graphs = {"compound": [], "solvent": []}

            manager.generate_graph("compound")
            manager.generate_graph("solvent")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                report_path = manager.write_graph_error_report()

            self.assertEqual(Path(report_path), model_path / "graph_error.csv")
            self.assertFalse((root / "graph_error.csv").exists())
            report = pd.read_csv(report_path)
            self.assertEqual(report.columns.tolist(), ["smiles", "type", "reason", "row_index"])
            self.assertEqual(report["type"].tolist(), ["compound", "solvent"])
            self.assertEqual(report["row_index"].tolist(), [101, 100])
            message = "\n".join(str(item.message) for item in caught)
            self.assertIn("2 unique CSV rows", message)
            self.assertIn("compound", message)
            self.assertIn(str(model_path / "graph_error.csv"), message)


if __name__ == "__main__":
    unittest.main()
