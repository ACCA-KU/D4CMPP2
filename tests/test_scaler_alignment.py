import unittest

try:
    from .markers import heavy_test
except ImportError:
    from markers import heavy_test


class _SubsetDataset:
    def __init__(self, target):
        import torch
        self.target = torch.as_tensor(target, dtype=torch.float32)
        self.indices = None

    def __len__(self):
        return len(self.target)

    def get_subDataset(self, indices):
        subset = _SubsetDataset(self.target[indices])
        subset.indices = list(indices)
        return subset


class _Graph:
    def number_of_nodes(self):
        return 1


class TargetScalerAndAlignmentTests(unittest.TestCase):
    @staticmethod
    def _manager(targets, labels=None, scope="train"):
        import pandas as pd
        import torch
        from D4CMPP2.src.DataManager.MolDataManager import MolDataManager
        from D4CMPP2.src.utils.scaler import Scaler

        manager = object.__new__(MolDataManager)
        manager.config = {"DATA_PATH": "synthetic.csv", "target_scaler_fit_scope": scope}
        manager.random_seed = 42
        manager.target_value = torch.tensor(targets, dtype=torch.float32).reshape(-1, 1)
        manager.whole_dataset = _SubsetDataset(manager.target_value)
        manager.scaler = Scaler("standard")
        manager.set = None if labels is None else pd.Series(labels)
        return manager

    @heavy_test
    def test_explicit_split_fits_scaler_on_train_targets_only(self):
        manager = self._manager([0.0, 2.0, 100.0, 200.0], ["train", "train", "val", "test"])
        manager.split_data()
        self.assertEqual(manager.scaler.scaler.mean_.tolist(), [1.0])
        self.assertEqual(manager.train_dataset.indices, [0, 1])
        self.assertEqual(manager.val_dataset.indices, [2])
        self.assertEqual(manager.test_dataset.indices, [3])

    @heavy_test
    def test_automatic_split_keeps_indices_and_fits_train_only(self):
        import numpy as np
        from sklearn.model_selection import train_test_split

        targets = np.arange(10, dtype=float)
        manager = self._manager(targets)
        train_idx, test_idx = train_test_split(np.arange(10), test_size=0.1, random_state=42)
        train_idx, val_idx = train_test_split(train_idx, test_size=1 / 9, random_state=42)
        manager.split_data()
        self.assertEqual(manager.train_dataset.indices, train_idx.tolist())
        self.assertEqual(manager.val_dataset.indices, val_idx.tolist())
        self.assertEqual(manager.test_dataset.indices, test_idx.tolist())
        self.assertAlmostEqual(manager.scaler.scaler.mean_[0], targets[train_idx].mean())

    @heavy_test
    def test_all_scope_reproduces_full_data_statistics_with_warning(self):
        manager = self._manager([0.0, 2.0, 100.0], ["train", "train", "val"], scope="all")
        with self.assertWarnsRegex(UserWarning, r"fits target scaling on validation/test"):
            manager.split_data()
        self.assertAlmostEqual(manager.scaler.scaler.mean_[0], 34.0)

    def test_old_load_scope_is_wired(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        main_source = (root / "src" / "api" / "training.py").read_text(encoding="utf-8")
        self.assertIn("legacy_scaler_scope", main_source)
        self.assertIn("config['target_scaler_fit_scope'] = 'all'", main_source)


if __name__ == "__main__":
    unittest.main()
