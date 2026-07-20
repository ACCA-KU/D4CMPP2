import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def parse(relative_path):
    return ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))


def function(tree, name):
    return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)


class CurrentFailureContractTests(unittest.TestCase):
    def test_run_defaults_to_raise_and_keeps_explicit_legacy_none_path(self):
        run = function(parse("_main.py"), "run")
        handlers = [node for node in ast.walk(run) if isinstance(node, ast.ExceptHandler)]
        self.assertEqual(len(handlers), 2)
        by_name = {
            handler.type.id: handler
            for handler in handlers
            if isinstance(handler.type, ast.Name)
        }
        self.assertEqual(set(by_name), {"KeyboardInterrupt", "Exception"})
        self.assertTrue(any(isinstance(node, ast.Raise) for node in ast.walk(by_name["KeyboardInterrupt"])))
        exception_handler = by_name["Exception"]
        self.assertTrue(any(isinstance(node, ast.Raise) for node in ast.walk(exception_handler)))
        self.assertTrue(any(isinstance(node, ast.Return) and isinstance(node.value, ast.Constant) and node.value.value is None for node in ast.walk(exception_handler)))
        self.assertTrue(any(isinstance(node, ast.Constant) and node.value == "legacy_silent_errors" for node in ast.walk(exception_handler)))

        train_error_checks = [
            node for node in ast.walk(run)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Attribute)
            and node.test.left.attr == "train_error"
        ]
        self.assertEqual(len(train_error_checks), 1)
        self.assertTrue(any(isinstance(node, ast.Raise) for node in ast.walk(train_error_checks[0])))

    def test_postprocessor_propagates_by_default_and_has_explicit_legacy_paths(self):
        postprocess = function(parse("src/PostProcessor.py"), "postprocess")
        handlers = [node for node in ast.walk(postprocess) if isinstance(node, ast.ExceptHandler)]
        self.assertEqual(len(handlers), 3)
        self.assertTrue(all(isinstance(handler.type, ast.Name) and handler.type.id == "Exception" for handler in handlers))
        for handler in handlers:
            self.assertTrue(any(isinstance(node, ast.Raise) for node in ast.walk(handler)))
            self.assertTrue(any(isinstance(node, ast.Constant) and node.value == "legacy_silent_errors" for node in ast.walk(handler)))

    def test_trainer_preserves_original_exception_traceback(self):
        train = function(parse("src/TrainManager/TrainManager.py"), "train")
        handlers = [node for node in ast.walk(train) if isinstance(node, ast.ExceptHandler)]
        exception_handler = next(
            handler for handler in handlers
            if isinstance(handler.type, ast.Name) and handler.type.id == "Exception"
        )
        raises = [node for node in exception_handler.body if isinstance(node, ast.Raise)]
        self.assertEqual(len(raises), 1)
        self.assertIsNone(raises[0].exc, "Use bare raise to preserve the original exception type and traceback")

    def test_keyboard_interrupt_and_grid_search_control_flow_are_not_swallowed(self):
        trainer = function(parse("src/TrainManager/TrainManager.py"), "train")
        keyboard_handler = next(
            handler for handler in ast.walk(trainer)
            if isinstance(handler, ast.ExceptHandler)
            and isinstance(handler.type, ast.Name)
            and handler.type.id == "KeyboardInterrupt"
        )
        self.assertTrue(any(isinstance(node, ast.Raise) for node in ast.walk(keyboard_handler)))
        self.assertTrue(any(isinstance(node, ast.Constant) and node.value == "legacy_silent_errors" for node in ast.walk(keyboard_handler)))

        grid = function(parse("grid_search.py"), "grid_search")
        handlers = [node for node in ast.walk(grid) if isinstance(node, ast.ExceptHandler)]
        self.assertEqual(
            [handler.type.id for handler in handlers if isinstance(handler.type, ast.Name)],
            ["KeyboardInterrupt", "Exception"],
        )


if __name__ == "__main__":
    unittest.main()
