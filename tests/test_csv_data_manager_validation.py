from pathlib import Path
import tempfile
import unittest

try:
    from .markers import heavy_test
except ImportError:
    from markers import heavy_test


class _IdentityScaler:
    def fit_transform(self, values):
        return values


class CsvDataManagerIntegrationTests(unittest.TestCase):
    @staticmethod
    def _manager(path, target=None, molecule=None, numeric=None):
        from D4CMPP2.src.DataManager.MolDataManager import MolDataManager

        manager = object.__new__(MolDataManager)
        manager.config = {"DATA_PATH": str(path)}
        manager.target = target or ["target"]
        manager.molecule_columns = molecule or ["compound"]
        manager.numeric_input_columns = numeric or []
        manager.scaler = _IdentityScaler()
        manager._molecule_smiles = {}
        manager.molecule_smiles = {}
        manager.valid_smiles = {}
        return manager

    @heavy_test
    def test_valid_utf8_and_cp949_files_load_with_aligned_arrays(self):
        with tempfile.TemporaryDirectory(prefix="d4cmpp2-csv-") as temporary:
            root = Path(temporary)
            utf8 = root / "valid.csv"
            utf8.write_text(
                "compound,target,numeric,set\nCC,1.0,10,train\nCCC,2.0,20,val\n",
                encoding="utf-8",
            )
            manager = self._manager(utf8, numeric=["numeric"])
            manager.load_data()
            self.assertEqual(len(manager._molecule_smiles["compound"]), 2)
            self.assertEqual(len(manager.target_value), 2)
            self.assertEqual(len(manager.set), 2)
            self.assertEqual(manager.numeric_inputs["numeric"].tolist(), [10.0, 20.0])

            cp949 = root / "cp949.csv"
            cp949.write_bytes("compound,용해도,set\nCC,1.0,train\nCCC,2.0,val\n".encode("cp949"))
            manager = self._manager(cp949, target=["용해도"])
            manager.load_data()
            self.assertEqual(manager.target_value.shape, (2, 1))

    @heavy_test
    def test_schema_numeric_target_and_split_fail_before_graph_generation(self):
        cases = (
            (
                "missing.csv",
                "compound,target,set\nCC,1.0,train\nCCC,2.0,val\n",
                {"molecule": ["compound", "solvent"]},
                r"missing\.csv.*molecule.*solvent.*Available columns",
            ),
            (
                "numeric.csv",
                "compound,target,numeric,set\nCC,1.0,bad,train\nCCC,2.0,2,val\n",
                {"numeric": ["numeric"]},
                r"Numeric input.*numeric.*\[0\]",
            ),
            (
                "target.csv",
                "compound,target,set\nCC,bad,train\nCCC,2.0,val\n",
                {},
                r"Target.*target.*\[0\]",
            ),
            (
                "nan.csv",
                "compound,target,set\nCC,,train\nCCC,,val\n",
                {},
                r"target.*only NaN",
            ),
            (
                "labels.csv",
                "compound,target,set\nCC,1.0,train\nCCC,2.0,validation\n",
                {},
                r"invalid labels.*validation.*Allowed labels",
            ),
            (
                "split.csv",
                "compound,target,set\nCC,1.0,train\nCCC,2.0,train\n",
                {},
                r"missing required.*val.*Split counts",
            ),
        )
        with tempfile.TemporaryDirectory(prefix="d4cmpp2-csv-") as temporary:
            root = Path(temporary)
            for filename, content, options, pattern in cases:
                with self.subTest(filename=filename):
                    path = root / filename
                    path.write_text(content, encoding="utf-8")
                    manager = self._manager(path, **options)
                    with self.assertRaisesRegex(ValueError, pattern):
                        manager.load_data()

    @heavy_test
    def test_empty_and_too_small_automatic_split_files_are_distinguished(self):
        with tempfile.TemporaryDirectory(prefix="d4cmpp2-csv-") as temporary:
            root = Path(temporary)
            empty = root / "empty.csv"
            empty.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"empty\.csv.*empty or has no columns"):
                self._manager(empty).load_data()

            small = root / "small.csv"
            small.write_text("compound,target\nCC,1.0\nCCC,2.0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"small\.csv.*2 usable rows.*at least 10"):
                self._manager(small).load_data()


if __name__ == "__main__":
    unittest.main()
