from types import SimpleNamespace
import unittest

try:
    from .fixtures import load_source_module
except ImportError:
    from fixtures import load_source_module


validation = load_source_module("src/utils/config_validation.py", "runtime_validation_for_tests")


class FakeCuda:
    def __init__(self, available=False, count=0):
        self.available = available
        self.count = count

    def is_available(self):
        return self.available

    def device_count(self):
        return self.count


def fake_torch(available=False, count=0, version="2.test"):
    return SimpleNamespace(__version__=version, cuda=FakeCuda(available, count))


def importer_with_backend(name):
    if name == "torch_geometric":
        return SimpleNamespace(__version__="2.test")
    raise ImportError(name)


class RuntimeEnvironmentTests(unittest.TestCase):
    def test_cpu_pyg_environment_reports_versions_without_mutating_config(self):
        config = {"device": "cpu", "pin_memory": False}
        original = dict(config)

        versions = validation.validate_runtime_environment(
            config, torch_module=fake_torch(), importer=importer_with_backend
        )

        self.assertEqual(versions, {"torch": "2.test", "pyg": "2.test"})
        self.assertEqual(config, original)

    def test_unavailable_cuda_fails_with_cpu_remedy_and_no_automatic_fallback(self):
        config = {"device": "cuda:0", "pin_memory": False}
        with self.assertRaisesRegex(RuntimeError, r"cuda:0.*CUDA is not available.*device='cpu'.*does not switch"):
            validation.validate_runtime_environment(
                config, torch_module=fake_torch(), importer=importer_with_backend
            )
        self.assertEqual(config["device"], "cuda:0")

    def test_out_of_range_cuda_index_reports_count_and_valid_indices(self):
        with self.assertRaisesRegex(RuntimeError, r"cuda:2.*2 CUDA device.*0\.\.1"):
            validation.validate_runtime_environment(
                {"device": "cuda:2", "pin_memory": False},
                torch_module=fake_torch(available=True, count=2),
                importer=importer_with_backend,
            )

    def test_backend_import_failure_preserves_cause_and_versions(self):
        def failing_importer(name):
            raise ImportError("missing compiled graph library")

        with self.assertRaisesRegex(RuntimeError, r"pyg.*PyTorch '2\.test'.*torch_geometric.*missing compiled") as caught:
            validation.validate_runtime_environment(
                {"device": "cpu", "pin_memory": False},
                torch_module=fake_torch(),
                importer=failing_importer,
            )
        self.assertIsInstance(caught.exception.__cause__, ImportError)

    def test_explicit_pyg_backend_reports_version(self):
        versions = validation.validate_runtime_environment(
            {"device": "cpu", "pin_memory": False},
            backend="pyg",
            torch_module=fake_torch(),
            importer=importer_with_backend,
        )
        self.assertEqual(versions["pyg"], "2.test")

    def test_cpu_pin_memory_warns_but_keeps_explicit_value(self):
        config = {"device": "cpu", "pin_memory": True}
        with self.assertWarnsRegex(UserWarning, r"CPU-only training.*pin_memory=False"):
            validation.validate_runtime_environment(
                config, torch_module=fake_torch(), importer=importer_with_backend
            )
        self.assertTrue(config["pin_memory"])


if __name__ == "__main__":
    unittest.main()
