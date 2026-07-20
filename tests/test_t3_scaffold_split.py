import json
import tempfile
import unittest
from pathlib import Path

from fixtures import isolated_workdir
from markers import heavy_test


SCAFFOLD_SMILES = [
    "c1ccccc1",
    "Cc1ccccc1",
    "c1ccncc1",
    "c1ccoc1",
    "c1ccsc1",
    "C1CCCCC1",
    "C1CCCC1",
    "c1ccc2ccccc2c1",
    "c1ccc2[nH]ccc2c1",
    "C1CCOC1",
    "C1CCNC1",
    "c1ncc[nH]1",
]


class ScaffoldSplitTests(unittest.TestCase):
    @heavy_test
    def test_scaffold_groups_are_reproducible_and_never_cross_splits(self):
        from D4CMPP2.src.utils.splitting import (
            murcko_scaffold,
            scaffold_split_indices,
        )

        first, scaffolds = scaffold_split_indices(SCAFFOLD_SMILES, seed=73)
        second, _ = scaffold_split_indices(SCAFFOLD_SMILES, seed=73)
        self.assertEqual(
            [indices.tolist() for indices in first],
            [indices.tolist() for indices in second],
        )
        sets = [
            {scaffolds[index] for index in indices} for indices in first
        ]
        self.assertFalse(sets[0] & sets[1])
        self.assertFalse(sets[0] & sets[2])
        self.assertFalse(sets[1] & sets[2])
        self.assertTrue(
            murcko_scaffold(SCAFFOLD_SMILES[0])
            == murcko_scaffold(SCAFFOLD_SMILES[1])
        )
        self.assertTrue(all(len(indices) > 0 for indices in first))

    @heavy_test
    def test_too_few_scaffolds_and_missing_predefined_column_are_actionable(self):
        import torch

        from D4CMPP2.src.DataManager.MolDataManager import MolDataManager
        from D4CMPP2.src.utils.splitting import scaffold_split_indices

        with self.assertRaisesRegex(ValueError, "at least 3 distinct"):
            scaffold_split_indices(["CC", "CCC", "CCCC"], seed=42)

        manager = object.__new__(MolDataManager)
        manager.config = {
            "DATA_PATH": "synthetic.csv",
            "split_strategy": "predefined",
        }
        manager.set = None
        manager.random_seed = 42
        manager.target_value = torch.arange(10, dtype=torch.float32).reshape(-1, 1)

        class Dataset:
            target = manager.target_value

            def __len__(self):
                return 10

        manager.whole_dataset = Dataset()
        with self.assertRaisesRegex(ValueError, "requires a CSV 'set' column"):
            manager.split_data()

    @heavy_test
    def test_cpu_training_writes_scaffold_split_report_without_leakage(self):
        from D4CMPP2 import train

        with isolated_workdir() as root:
            data_path = root / "scaffolds.csv"
            rows = ["compound,target"]
            rows.extend(
                f"{smiles},{index / 10}"
                for index, smiles in enumerate(SCAFFOLD_SMILES)
            )
            data_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            (root / "models").mkdir()
            (root / "graphs").mkdir()
            model_path = root / "model"
            result = train(
                data=str(data_path),
                target=["target"],
                network="GCN",
                MODEL_PATH=str(model_path),
                MODEL_DIR=str(root / "models"),
                GRAPH_DIR=str(root / "graphs"),
                device="cpu",
                pin_memory=False,
                split_strategy="scaffold",
                split_random_seed=73,
                random_seed=73,
                max_epoch=1,
                batch_size=4,
                hidden_dim=8,
                conv_layers=2,
                linear_layers=2,
                dropout=0.1,
                save_prediction=False,
            )

            report = json.loads(
                (Path(result) / "split_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["strategy"], "scaffold")
            self.assertEqual(report["scaffold_overlap_count"], 0)
            self.assertEqual(sum(report["counts"].values()), len(SCAFFOLD_SMILES))
            self.assertTrue(all(value > 0 for value in report["counts"].values()))
            self.assertTrue((Path(result) / "split_assignments.csv").is_file())


if __name__ == "__main__":
    unittest.main()
