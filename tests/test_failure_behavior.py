import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

try:
    from .markers import heavy_test
except ImportError:
    from markers import heavy_test


class _TrainingFailure(RuntimeError):
    pass


class _FailingDataManager:
    unwrapper = None

    def __init__(self, config):
        self.config = config

    def init_data(self):
        raise _TrainingFailure("data initialization failed")


class _UnusedManager:
    def __init__(self, *args, **kwargs):
        pass


class _FailingNetworkManager:
    network = object()

    def save_params(self, path):
        raise OSError("final model write failed")


class _WritingNetworkManager:
    network = object()

    def __init__(self, payload=b"new-final"):
        self.payload = payload

    def save_params(self, path):
        Path(path).write_bytes(self.payload)


class _PartialFailingNetworkManager:
    network = object()

    def save_params(self, path):
        Path(path).write_bytes(b"partial")
        raise OSError("staging write failed")


class FailureBehaviorTests(unittest.TestCase):
    @staticmethod
    def _run_with_failing_data(config):
        from D4CMPP2 import _main

        with mock.patch.object(_main.module_loader, "load_data_manager", return_value=_FailingDataManager), \
             mock.patch.object(_main.module_loader, "load_network_manager", return_value=_UnusedManager), \
             mock.patch.object(_main.module_loader, "load_train_manager", return_value=_UnusedManager):
            return _main.run(config)

    @heavy_test
    def test_run_raises_original_exception_by_default_and_legacy_returns_none(self):
        with tempfile.TemporaryDirectory(prefix="d4cmpp2-failure-") as temporary:
            default_config = {"MODEL_PATH": temporary}
            with self.assertRaisesRegex(_TrainingFailure, "data initialization failed"):
                self._run_with_failing_data(default_config)

            legacy_config = {"MODEL_PATH": temporary, "legacy_silent_errors": True}
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = self._run_with_failing_data(legacy_config)
            self.assertIsNone(result)
            self.assertIn("_TrainingFailure: data initialization failed", output.getvalue())

    @heavy_test
    def test_postprocessor_model_save_failure_obeys_default_and_legacy_policy(self):
        from D4CMPP2.src.PostProcessor import PostProcessor

        with tempfile.TemporaryDirectory(prefix="d4cmpp2-failure-") as temporary:
            model_path = Path(temporary)
            processor = PostProcessor({"MODEL_PATH": str(model_path)})
            with self.assertRaisesRegex(OSError, "final model write failed"):
                processor.postprocess(None, _FailingNetworkManager(), None, None, None, None)

            legacy = PostProcessor({"MODEL_PATH": str(model_path), "legacy_silent_errors": True})
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = legacy.postprocess(None, _FailingNetworkManager(), None, None, None, None)
            self.assertIsNone(result)
            self.assertIn("Model saving failed", output.getvalue())
            self.assertIn("OSError: final model write failed", output.getvalue())

    @heavy_test
    def test_successful_final_commit_only_removes_numeric_package_checkpoints(self):
        from D4CMPP2.src.PostProcessor import PostProcessor

        with tempfile.TemporaryDirectory(prefix="d4cmpp2-checkpoint-") as temporary:
            model_path = Path(temporary)
            (model_path / "final.pth").write_bytes(b"old-final")
            (model_path / "param_0.125.pth").write_bytes(b"best")
            (model_path / "param_1e-05.pth").write_bytes(b"other")
            (model_path / "param_backup.pth").write_bytes(b"user")
            (model_path / "experiment_backup.pth").write_bytes(b"user")
            processor = PostProcessor({"MODEL_PATH": str(model_path)})

            processor.save_final_model(_WritingNetworkManager())

            self.assertEqual((model_path / "final.pth").read_bytes(), b"new-final")
            self.assertFalse((model_path / "param_0.125.pth").exists())
            self.assertFalse((model_path / "param_1e-05.pth").exists())
            self.assertEqual((model_path / "param_backup.pth").read_bytes(), b"user")
            self.assertEqual((model_path / "experiment_backup.pth").read_bytes(), b"user")
            self.assertEqual(list(model_path.glob(".final.pth.*.tmp")), [])

    @heavy_test
    def test_staging_save_failure_preserves_previous_final_and_best_checkpoint(self):
        from D4CMPP2.src.PostProcessor import PostProcessor

        with tempfile.TemporaryDirectory(prefix="d4cmpp2-checkpoint-") as temporary:
            model_path = Path(temporary)
            (model_path / "final.pth").write_bytes(b"old-final")
            (model_path / "param_0.125.pth").write_bytes(b"best")
            processor = PostProcessor({"MODEL_PATH": str(model_path)})

            with self.assertRaisesRegex(OSError, "staging write failed"):
                processor.save_final_model(_PartialFailingNetworkManager())

            self.assertEqual((model_path / "final.pth").read_bytes(), b"old-final")
            self.assertEqual((model_path / "param_0.125.pth").read_bytes(), b"best")
            self.assertEqual(list(model_path.glob(".final.pth.*.tmp")), [])

    @heavy_test
    def test_atomic_replace_failure_preserves_previous_final_and_checkpoints(self):
        from D4CMPP2.src.PostProcessor import PostProcessor

        with tempfile.TemporaryDirectory(prefix="d4cmpp2-checkpoint-") as temporary:
            model_path = Path(temporary)
            (model_path / "final.pth").write_bytes(b"old-final")
            (model_path / "param_0.125.pth").write_bytes(b"best")
            processor = PostProcessor({"MODEL_PATH": str(model_path)})

            with mock.patch("D4CMPP2.src.PostProcessor.os.replace", side_effect=OSError("replace failed")), \
                 self.assertRaisesRegex(OSError, "replace failed"):
                processor.save_final_model(_WritingNetworkManager())

            self.assertEqual((model_path / "final.pth").read_bytes(), b"old-final")
            self.assertEqual((model_path / "param_0.125.pth").read_bytes(), b"best")
            self.assertEqual(list(model_path.glob(".final.pth.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
