import importlib
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest

from fixtures import ROOT
from markers import heavy_test


class ImportDiscoverySourceTests(unittest.TestCase):
    def test_package_root_does_not_mutate_sys_path(self):
        source = (ROOT / "__init__.py").read_text(encoding="utf-8")
        self.assertNotIn("sys.path.append", source)

    def test_manager_packages_do_not_scan_directories(self):
        paths = [
            ROOT / "src" / "TrainManager" / "__init__.py",
            ROOT / "src" / "DataManager" / "__init__.py",
            ROOT / "src" / "NetworkManager" / "__init__.py",
            ROOT / "src" / "DataManager" / "GraphGenerator" / "__init__.py",
        ]
        for path in paths:
            with self.subTest(path=path):
                source = path.read_text(encoding="utf-8")
                self.assertNotIn("pkgutil.iter_modules", source)


@heavy_test
class ImportDiscoveryRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        package_parent = str(ROOT.parent)
        if package_parent not in sys.path:
            sys.path.insert(0, package_parent)
        cls.loader = importlib.import_module("D4CMPP2.src.utils.module_loader")

    def test_builtin_loader_imports_configured_module(self):
        trainer = self.loader.load_default_train_manager(
            "TrainManager", "Trainer"
        )
        self.assertEqual(trainer.__name__, "Trainer")

        package = importlib.import_module("D4CMPP2.src.TrainManager")
        legacy_module = package.TrainManager
        self.assertIs(legacy_module.Trainer, trainer)

    def test_custom_loader_restores_path_and_supports_sibling_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "helper.py").write_text("VALUE = 17\n", encoding="utf-8")
            (path / "manager.py").write_text(
                textwrap.dedent(
                    """
                    from helper import VALUE

                    class Manager:
                        value = VALUE
                    """
                ),
                encoding="utf-8",
            )
            before = list(sys.path)
            manager = self.loader.load_module(temporary, "manager", "Manager")
            self.assertEqual(manager.value, 17)
            self.assertEqual(sys.path, before)

    def test_same_custom_module_name_from_different_paths_is_isolated(self):
        classes = []
        before = list(sys.path)
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            for directory, value in ((first, 1), (second, 2)):
                Path(directory, "manager.py").write_text(
                    f"class Manager:\n    value = {value}\n",
                    encoding="utf-8",
                )
                classes.append(
                    self.loader.load_module(directory, "manager", "Manager")
                )
        self.assertEqual([cls.value for cls in classes], [1, 2])
        self.assertIsNot(classes[0], classes[1])
        self.assertEqual(sys.path, before)


if __name__ == "__main__":
    unittest.main()
