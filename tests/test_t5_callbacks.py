import tempfile
import unittest
from pathlib import Path

from fixtures import load_source_module
from markers import heavy_test

callbacks_module = load_source_module(
    "src/TrainManager/callbacks.py",
    "t5_callbacks_contract",
)

class CallbackContractTests(unittest.TestCase):
    def test_dispatch_order_and_immutable_snapshot(self):
        CallbackDispatcher = callbacks_module.CallbackDispatcher
        EventHistory = callbacks_module.EventHistory
        TrainingEvent = callbacks_module.TrainingEvent

        calls = []

        class First:
            def on_validation_end(self, event):
                calls.append(("first-specific", event.name))

            def on_event(self, event):
                calls.append(("first-generic", event.name))

        class Second:
            def on_event(self, event):
                calls.append(("second", event.name))

        event = TrainingEvent(
            "validation_end",
            epoch=2,
            checkpoint_paths={"best": "best.ckpt"},
        )
        CallbackDispatcher([First(), Second()]).emit(event)

        self.assertEqual(
            calls,
            [
                ("first-specific", "validation_end"),
                ("first-generic", "validation_end"),
                ("second", "validation_end"),
            ],
        )
        with self.assertRaises(TypeError):
            event.checkpoint_paths["latest"] = "latest.ckpt"
        with self.assertRaisesRegex(TypeError, "callbacks must be a sequence"):
            CallbackDispatcher("not-a-sequence-of-callbacks")
        history = EventHistory()
        CallbackDispatcher([history]).emit(event)
        self.assertEqual(history.events, [event])

    @heavy_test
    def test_training_events_preserve_scheduler_order_and_empty_callback_behavior(self):
        import sys

        from fixtures import ROOT

        if str(ROOT.parent) not in sys.path:
            sys.path.insert(0, str(ROOT.parent))
        from D4CMPP2.src.TrainManager.TrainManager import Trainer

        class FixtureTrainer(Trainer):
            def train_epoch(self, network_manager, train_loader):
                network_manager.operations.append("train")
                return 2.0

            def evaluate(self, network_manager, val_loader):
                network_manager.operations.append("validation")
                return 3.0

        class FixtureManager:
            next_epoch = 0

            def __init__(self):
                self.operations = []

            def get_lr(self):
                return 0.01

            def scheduler_step(self, val_loss, completed_epoch):
                self.operations.append(("scheduler", val_loss, completed_epoch))
                return False

            def load_best_checkpoint(self):
                self.operations.append("load_best")

        class Recorder:
            def __init__(self, operations):
                self.operations = operations
                self.events = []

            def on_event(self, event):
                self.events.append(event)
                self.operations.append(("event", event.name))

        with tempfile.TemporaryDirectory(prefix="d4cmpp2-t5-callback-") as temporary:
            manager = FixtureManager()
            recorder = Recorder(manager.operations)
            trainer = FixtureTrainer(
                {
                    "MODEL_PATH": temporary,
                    "device": "cpu",
                    "max_epoch": 1,
                }
            )
            trainer.set_callbacks([recorder])
            trainer.train(manager, (), ())

            self.assertEqual(
                [event.name for event in recorder.events],
                [
                    "run_start",
                    "epoch_start",
                    "train_epoch_end",
                    "validation_end",
                    "epoch_end",
                    "run_end",
                ],
            )
            self.assertLess(
                manager.operations.index(("event", "validation_end")),
                manager.operations.index(("scheduler", 3.0, 0)),
            )
            self.assertLess(
                manager.operations.index(("scheduler", 3.0, 0)),
                manager.operations.index(("event", "epoch_end")),
            )
            self.assertLess(
                manager.operations.index("load_best"),
                manager.operations.index(("event", "run_end")),
            )
            self.assertEqual(recorder.events[-2].train_loss, 2.0)
            self.assertEqual(recorder.events[-2].val_loss, 3.0)
            self.assertTrue((Path(temporary) / "learning_curve.csv").is_file())

            no_callback_manager = FixtureManager()
            no_callback_trainer = FixtureTrainer(
                {
                    "MODEL_PATH": temporary,
                    "device": "cpu",
                    "max_epoch": 1,
                }
            )
            no_callback_trainer.train(no_callback_manager, (), ())
            self.assertEqual(
                no_callback_manager.operations,
                ["train", "validation", ("scheduler", 3.0, 0), "load_best"],
            )

    @heavy_test
    def test_callback_failure_propagates_and_emits_exception_event(self):
        import sys

        from fixtures import ROOT

        if str(ROOT.parent) not in sys.path:
            sys.path.insert(0, str(ROOT.parent))
        from D4CMPP2.src.TrainManager.TrainManager import Trainer

        class FixtureTrainer(Trainer):
            def train_epoch(self, network_manager, train_loader):
                return 1.0

            def evaluate(self, network_manager, val_loader):
                return 1.0

        class FixtureManager:
            next_epoch = 0

            def get_lr(self):
                return 0.1

        class FailingCallback:
            def __init__(self):
                self.names = []

            def on_event(self, event):
                self.names.append(event.name)
                if event.name == "epoch_start":
                    raise RuntimeError("callback failed")

        with tempfile.TemporaryDirectory(prefix="d4cmpp2-t5-callback-error-") as temporary:
            callback = FailingCallback()
            trainer = FixtureTrainer(
                {
                    "MODEL_PATH": temporary,
                    "device": "cpu",
                    "max_epoch": 1,
                }
            )
            trainer.set_callbacks([callback])
            with self.assertRaisesRegex(RuntimeError, "callback failed"):
                trainer.train(FixtureManager(), (), ())
            self.assertEqual(
                callback.names,
                ["run_start", "epoch_start", "exception"],
            )
            self.assertIsInstance(trainer.train_error, RuntimeError)


if __name__ == "__main__":
    unittest.main()
