import unittest

from fixtures import load_source_module
from markers import heavy_test


contracts = load_source_module(
    "src/DataManager/contracts.py",
    "t5_data_manager_contracts",
)


class ManagerContractUnitTests(unittest.TestCase):
    def test_generalized_key_matrix_is_explicit(self):
        expected = contracts.GENERAL_BATCH_CONTRACT.expected_keys(
            molecule_columns=("compound", "solvent"),
            numeric_input_columns=("temperature",),
        )
        self.assertEqual(expected[0], "target")
        for key in (
            "compound_graphs",
            "compound_node_feature",
            "compound_edge_feature",
            "compound_smiles",
            "solvent_graphs",
            "solvent_smiles",
            "temperature_var",
        ):
            self.assertIn(key, expected)
        self.assertEqual(
            contracts.GENERAL_BATCH_CONTRACT.optional_keys,
            ("original_row_index",),
        )

    def test_metadata_free_custom_manager_is_allowed(self):
        class CustomDataset:
            @staticmethod
            def unwrapper(**batch):
                return batch

        class CustomManager:
            dataset = CustomDataset
            unwrapper = CustomDataset.unwrapper

        self.assertIsNone(
            contracts.validate_data_manager_contract(CustomManager(), {})
        )

    def test_missing_dimension_error_identifies_manager_and_keys(self):
        class BuiltinDataset:
            batch_contract = contracts.GENERAL_BATCH_CONTRACT

            @staticmethod
            def unwrapper(**batch):
                return batch

        class BrokenManager:
            dataset = BuiltinDataset
            unwrapper = BuiltinDataset.unwrapper
            feature_dimension_keys = ("node_dim", "edge_dim")
            graph_type = "mol"

        with self.assertRaisesRegex(
            ValueError,
            "BrokenManager.*edge_dim.*before NetworkManager.*Available config keys",
        ):
            contracts.validate_data_manager_contract(
                BrokenManager(),
                {"node_dim": 1},
            )

    def test_contract_snapshot_is_immutable(self):
        class BuiltinDataset:
            batch_contract = contracts.GENERAL_BATCH_CONTRACT

            @staticmethod
            def unwrapper(**batch):
                return batch

        class Manager:
            dataset = BuiltinDataset
            unwrapper = BuiltinDataset.unwrapper
            feature_dimension_keys = ("node_dim", "edge_dim")
            graph_type = "mol"

        snapshot = contracts.validate_data_manager_contract(
            Manager(),
            {"node_dim": 2, "edge_dim": 3},
        )
        with self.assertRaises(TypeError):
            snapshot.feature_dimensions["node_dim"] = 5


class BuiltinManagerContractTests(unittest.TestCase):
    @heavy_test
    def test_builtin_dataset_metadata_and_isa_compatibility_aliases(self):
        import sys

        from fixtures import ROOT

        if str(ROOT.parent) not in sys.path:
            sys.path.insert(0, str(ROOT.parent))

        from D4CMPP2.src.DataManager.Dataset.GraphDataset import (
            GraphDataset,
            GraphDataset_legacy,
            GraphDataset_withSolv,
        )
        from D4CMPP2.src.DataManager.Dataset.ISAGraphDataset import (
            ISAGraphDataset,
            ISAGraphDataset_legacy,
        )
        from D4CMPP2.src.NetworkManager.ISANetworkManager import ISANetworkManager
        from D4CMPP2.src.NetworkManager.NetworkManager import NetworkManager
        from D4CMPP2.src.TrainManager.ISATrainManager import ISATrainer
        from D4CMPP2.src.TrainManager.TrainManager import Trainer

        self.assertEqual(GraphDataset.batch_contract.name, "generalized_graph")
        self.assertEqual(GraphDataset_legacy.batch_contract.name, "legacy_general")
        self.assertEqual(GraphDataset_withSolv.batch_contract.name, "legacy_solvent")
        self.assertEqual(ISAGraphDataset.batch_contract.name, "generalized_isa")
        self.assertEqual(ISAGraphDataset_legacy.batch_contract.name, "legacy_isa")
        self.assertTrue(issubclass(ISANetworkManager, NetworkManager))
        self.assertTrue(issubclass(ISATrainer, Trainer))
        self.assertIn("Compatibility extension point", ISANetworkManager.__doc__)
        self.assertIn("Compatibility extension point", ISATrainer.__doc__)


if __name__ == "__main__":
    unittest.main()
