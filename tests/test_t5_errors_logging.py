import io
import logging
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fixtures import load_source_module
from markers import heavy_test


errors = load_source_module("src/api/errors.py", "t5_exception_contract")
output = load_source_module("src/utils/output.py", "t5_output_contract")


class ExceptionAndOutputUnitTests(unittest.TestCase):
    def test_exception_categories_keep_builtin_compatibility(self):
        cases = (
            (errors.ConfigError("config"), ValueError),
            (errors.DataError("data"), ValueError),
            (errors.GraphError("graph"), ValueError),
            (errors.ModelError("model"), RuntimeError),
            (errors.DependencyError("dependency"), RuntimeError),
            (errors.CheckpointNotFoundError("missing"), FileNotFoundError),
            (errors.CheckpointFormatError("format"), ValueError),
            (errors.CheckpointLoadError("load"), RuntimeError),
            (errors.CheckpointIOError("io"), OSError),
            (errors.ModuleLoadError("module"), ImportError),
            (errors.ManagerNotFoundError("manager"), FileNotFoundError),
        )
        for exception, builtin in cases:
            with self.subTest(exception=type(exception).__name__):
                self.assertIsInstance(exception, errors.D4CMPPError)
                self.assertIsInstance(exception, builtin)

    def test_output_adapter_preserves_stdout_and_supports_debug_logging(self):
        stream = io.StringIO()
        records = []

        class Handler(logging.Handler):
            def emit(self, record):
                records.append(record)

        logger = logging.getLogger("D4CMPP2.test.output")
        logger.handlers = [Handler()]
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        adapter = output.OutputAdapter(stream=stream, diagnostic_logger=logger)
        adapter.info("Device:", "cpu")
        adapter.debug("resolved %s", "config")

        self.assertEqual(stream.getvalue(), "Device: cpu\n")
        self.assertEqual(records[0].getMessage(), "resolved config")
        self.assertTrue(
            any(
                isinstance(handler, logging.NullHandler)
                for handler in output.logger.handlers
            )
        )


class ExceptionBoundaryIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys

        from fixtures import ROOT

        if str(ROOT.parent) not in sys.path:
            sys.path.insert(0, str(ROOT.parent))

    @heavy_test
    def test_custom_module_missing_syntax_and_body_errors_are_distinguished(self):
        from D4CMPP2.exceptions import (
            ManagerNotFoundError,
            ModuleLoadError,
        )
        from D4CMPP2.src.utils import module_loader

        with tempfile.TemporaryDirectory(prefix="d4cmpp2-module-errors-") as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(
                ManagerNotFoundError,
                "missing.py.*was not found.*Manager_PATH",
            ):
                module_loader._load_manager(
                    "Manager_PATH",
                    "manager_module",
                    "manager_class",
                    lambda *_: object,
                )(
                    {
                        "Manager_PATH": temporary,
                        "manager_module": "missing",
                        "manager_class": "Manager",
                    }
                )

            (root / "syntax_error.py").write_text("class Manager(:\n", encoding="utf-8")
            with self.assertRaises(ModuleLoadError) as syntax_context:
                module_loader.load_module(temporary, "syntax_error", "Manager")
            self.assertIsInstance(syntax_context.exception.__cause__, SyntaxError)

            (root / "body_error.py").write_text(
                "raise RuntimeError('module body failed')\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "module body failed"):
                module_loader.load_module(temporary, "body_error", "Manager")

    @heavy_test
    def test_checkpoint_categories_preserve_builtin_types_and_causes(self):
        from D4CMPP2.exceptions import CheckpointIOError, CheckpointLoadError
        from D4CMPP2.src.utils import checkpointing

        with mock.patch.object(
            checkpointing.torch,
            "load",
            side_effect=OSError("read failed"),
        ):
            with self.assertRaises(CheckpointLoadError) as load_context:
                checkpointing.load_checkpoint("broken.ckpt", "cpu")
        self.assertIsInstance(load_context.exception, RuntimeError)
        self.assertIsInstance(load_context.exception.__cause__, OSError)

        with tempfile.TemporaryDirectory(prefix="d4cmpp2-checkpoint-errors-") as temporary:
            path = Path(temporary) / "latest.ckpt"
            with mock.patch.object(
                checkpointing.torch,
                "save",
                side_effect=OSError("write failed"),
            ):
                with self.assertRaises(CheckpointIOError) as save_context:
                    checkpointing.atomic_torch_save({}, path)
            self.assertIsInstance(save_context.exception, OSError)
            self.assertIsInstance(save_context.exception.__cause__, OSError)

    @heavy_test
    def test_dependency_error_is_runtime_compatible_and_chained(self):
        from D4CMPP2.exceptions import DependencyError
        from D4CMPP2.src.utils.config_validation import validate_runtime_environment

        with self.assertRaises(DependencyError) as context:
            validate_runtime_environment(
                {"device": "cpu"},
                importer=lambda name: (_ for _ in ()).throw(
                    ImportError(f"{name} unavailable")
                ),
            )
        self.assertIsInstance(context.exception, RuntimeError)
        self.assertIsInstance(context.exception.__cause__, ImportError)


if __name__ == "__main__":
    unittest.main()
