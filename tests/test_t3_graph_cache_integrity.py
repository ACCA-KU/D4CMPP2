import tempfile
import unittest
from pathlib import Path
from unittest import mock
import shutil

from fixtures import ROOT
from markers import heavy_test


class GraphCacheIntegrityTests(unittest.TestCase):
    @heavy_test
    def test_fingerprint_tracks_smiles_explicit_h_and_isa_recipe(self):
        from D4CMPP2.src.DataManager.GraphGenerator.ISAGraphGenerator import ISAGraphGenerator
        from D4CMPP2.src.DataManager.MolDataManager import MolDataManager
        from D4CMPP2.src.utils.graph_cache import build_graph_recipe

        with tempfile.TemporaryDirectory(prefix="d4cmpp2-cache-v2-") as temporary:
            root = Path(temporary)
            rules = root / "functional_group.csv"
            shutil.copyfile(ROOT / "src" / "utils" / "functional_group.csv", rules)
            manager = object.__new__(MolDataManager)
            manager.data = str(root / "nested" / "dataset.csv")
            manager.graph_type = "img620"
            manager.explicit_h_columns = []
            manager.config = {
                "GRAPH_DIR": str(root),
                "MODEL_PATH": str(root),
                "FRAG_REF": str(rules),
                "max_dist": 4,
            }
            manager.gg = ISAGraphGenerator(str(rules), (6, 2, 0))
            manager._molecule_smiles = {"compound": ["CCO", "CC"]}

            first = build_graph_recipe(manager, "compound")
            manager._molecule_smiles["compound"] = ["CC", "CCO"]
            reordered = build_graph_recipe(manager, "compound")
            manager._molecule_smiles["compound"] = ["CCO", "CC"]
            manager.explicit_h_columns = ["compound"]
            explicit = build_graph_recipe(manager, "compound")
            manager.explicit_h_columns = []
            manager.config["max_dist"] = 5
            distance = build_graph_recipe(manager, "compound")
            manager.config["max_dist"] = 4
            rules.write_bytes(rules.read_bytes() + b"\n")
            changed_rules = build_graph_recipe(manager, "compound")

            self.assertEqual(len({
                first["fingerprint"],
                reordered["fingerprint"],
                explicit["fingerprint"],
                distance["fingerprint"],
                changed_rules["fingerprint"],
            }), 5)

    @heavy_test
    def test_atomic_failure_preserves_existing_cache_and_shape_is_validated(self):
        import torch

        from D4CMPP2.src.DataManager.GraphGenerator.MolGraphGenerator import MolGraphGenerator
        from D4CMPP2.src.DataManager.MolDataManager import MolDataManager
        from D4CMPP2.src.utils.graph_cache import atomic_save_graph_cache

        with tempfile.TemporaryDirectory(prefix="d4cmpp2-cache-v2-") as temporary:
            root = Path(temporary)
            destination = root / "cache.pt"
            torch.save({"old": True}, destination)
            with mock.patch("D4CMPP2.src.utils.graph_cache.os.replace", side_effect=OSError("blocked")):
                with self.assertRaisesRegex(OSError, "blocked"):
                    atomic_save_graph_cache({"new": True}, destination)
            self.assertEqual(torch.load(destination, weights_only=False), {"old": True})
            self.assertEqual(list(root.glob(".*.tmp")), [])

            generator = MolGraphGenerator()
            manager = object.__new__(MolDataManager)
            manager.data = "fixture"
            manager.graph_type = "mol"
            manager.explicit_h_columns = []
            manager.config = {"GRAPH_DIR": str(root)}
            manager.gg = generator
            manager._molecule_smiles = {"compound": ["CC"]}
            manager.molecule_graphs = {"compound": [generator.get_graph("CC")]}
            manager.graph_errors = []
            manager.original_row_indices = torch.tensor([0])
            manager.save_graphs("compound")
            path = Path(manager.get_graphs_path("compound"))
            payload = torch.load(path, weights_only=False)
            payload["graphs"][0].x = torch.zeros((2, generator.node_dim + 1))
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, r"graphs\[0\]\.x shape"):
                manager.load_graphs("compound")

            payload["graphs"][0] = generator.get_graph("CC")
            payload["graphs"][0].edge_attr = payload["graphs"][0].edge_attr[:-1]
            torch.save(payload, path)
            with self.assertRaisesRegex(
                ValueError,
                r"graphs\[0\] edge feature rows=1, but edge_index contains 2 edges",
            ):
                manager.load_graphs("compound")

    @heavy_test
    def test_explicit_legacy_policy_loads_v1_without_deleting_it(self):
        import torch

        from D4CMPP2.src.DataManager.GraphGenerator.MolGraphGenerator import MolGraphGenerator
        from D4CMPP2.src.DataManager.MolDataManager import MolDataManager
        from D4CMPP2.src.utils.graph_cache import legacy_cache_paths

        with tempfile.TemporaryDirectory(prefix="d4cmpp2-cache-v2-") as temporary:
            generator = MolGraphGenerator()
            manager = object.__new__(MolDataManager)
            manager.data = "fixture"
            manager.graph_type = "mol"
            manager.explicit_h_columns = []
            manager.molecule_columns = ["compound"]
            manager.config = {"GRAPH_DIR": temporary, "graph_cache_policy": "legacy"}
            manager.gg = generator
            manager._molecule_smiles = {"compound": ["CC"]}
            manager.molecule_graphs = {"compound": []}
            manager.graph_errors = []
            manager.original_row_indices = torch.tensor([0])
            legacy, _ = legacy_cache_paths(manager, "compound")
            graph = generator.get_graph("CC")
            torch.save({
                "graph_backend": "pyg",
                "graph_schema_version": 1,
                "smiles": ["CC"],
                "graphs": [graph],
            }, legacy)

            with self.assertWarnsRegex(RuntimeWarning, "legacy v1"):
                manager.prepare_graph()
            self.assertTrue(legacy.is_file())
            self.assertEqual(manager.molecule_graphs["compound"][0].num_nodes, 2)
