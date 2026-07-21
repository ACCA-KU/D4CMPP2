import ast
import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_FIELDS = (
    "name",
    "network",
    "data_manager_module",
    "data_manager_class",
    "network_manager_module",
    "network_manager_class",
    "train_manager_module",
    "train_manager_class",
    "description",
    "version",
)


def parse_module(relative_path):
    return ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))


def literal_assignment(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                return ast.literal_eval(node.value)
    raise AssertionError(f"Assignment {name!r} was not found")


def parse_registry():
    """Parse the registry's intentionally simple two-level scalar YAML shape."""
    registry = {}
    current = None
    for raw_line in (ROOT / "network_refer.yaml").read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if not raw_line.startswith((" ", "\t")) and raw_line.endswith(":"):
            current = raw_line[:-1]
            registry[current] = {}
            continue
        match = re.match(r"^\s+([^:]+):\s*(.*?)\s*$", raw_line)
        if match and current:
            key, value = match.groups()
            registry[current][key] = value.strip('"\'')
    return registry


def yaml_top_level_keys(path):
    return {
        match.group(1)
        for line in path.read_text(encoding="utf-8").splitlines()
        if (match := re.match(r"^([A-Za-z_][A-Za-z0-9_]*):", line))
    }


class PublicApiContractTests(unittest.TestCase):
    def test_top_level_exports(self):
        tree = parse_module("__init__.py")
        exports = {}
        for node in tree.body:
            if not isinstance(node, ast.ImportFrom):
                continue
            for alias in node.names:
                public_name = alias.asname or alias.name
                if public_name in {"train", "grid_search", "optimize", "compare_experiments", "Analyzer", "Segmentator", "Data"}:
                    exports[public_name] = (node.module, alias.name)

        self.assertEqual(
            exports,
            {
                "train": ("D4CMPP2._main", "train"),
                "grid_search": ("D4CMPP2.grid_search", "grid_search"),
                "optimize": ("D4CMPP2.optimize", "optimize"),
                "compare_experiments": ("D4CMPP2.src.utils.leaderboard", "compare_experiments"),
                "Analyzer": ("D4CMPP2.src.Analyzer", "Analyzer"),
                "Segmentator": ("D4CMPP2.src.utils.sculptor", "Segmentator"),
                "Data": ("D4CMPP2", "_Data"),
            },
        )

    def test_training_entry_point_signatures(self):
        main_tree = parse_module("_main.py")
        grid_tree = parse_module("grid_search.py")
        train = next(node for node in main_tree.body if isinstance(node, ast.FunctionDef) and node.name == "train")
        grid = next(node for node in grid_tree.body if isinstance(node, ast.FunctionDef) and node.name == "grid_search")

        self.assertEqual([arg.arg for arg in train.args.args], [])
        self.assertEqual(train.args.kwarg.arg, "kwargs")
        self.assertEqual([arg.arg for arg in grid.args.args], ["hyperparameters"])
        self.assertEqual(grid.args.kwarg.arg, "kwargs")

    def test_train_default_config(self):
        actual = literal_assignment(parse_module("_main.py"), "config0")
        expected = {
            "data": None,
            "target": None,
            "network": None,
            "scaler": "standard",
            "optimizer": "Adam",
            "max_epoch": 2000,
            "batch_size": 256,
            "learning_rate": 0.001,
            "weight_decay": 0.0005,
            "lr_patience": 80,
            "early_stopping_patience": 200,
            "min_lr": 1e-5,
            "device": "cuda:0",
            "pin_memory": False,
            "target_scaler_fit_scope": "train",
            "legacy_silent_errors": False,
            "random_seed": None,
            "deterministic_algorithms": False,
            "verbose": True,
            "hidden_dim": None,
            "conv_layers": None,
            "linear_layers": None,
            "dropout": None,
            "solv_hidden_dim": None,
            "solv_conv_layers": None,
            "solv_linear_layers": None,
        }
        self.assertEqual(actual, expected)

    def test_analyzers_use_current_trainer_prediction_contract(self):
        tree = parse_module("src/Analyzer/core.py")
        trainer_predict_assignments = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "predict"
            and isinstance(node.value.func.value, ast.Attribute)
            and isinstance(node.value.func.value.value, ast.Name)
            and node.value.func.value.value.id == "self"
            and node.value.func.value.attr == "tm"
        ]

        self.assertEqual(len(trainer_predict_assignments), 2)
        for assignment in trainer_predict_assignments:
            with self.subTest(line=assignment.lineno):
                self.assertEqual(len(assignment.targets), 1)
                self.assertIsInstance(assignment.targets[0], ast.Tuple)
                self.assertEqual(len(assignment.targets[0].elts), 3)


class RegistryContractTests(unittest.TestCase):
    def test_registry_snapshot(self):
        registry = parse_registry()
        snapshot = json.loads(
            (ROOT / "tests" / "snapshots" / "network_registry.json").read_text(encoding="utf-8")
        )
        normalized = {
            network_id: [config[field] for field in REGISTRY_FIELDS]
            for network_id, config in registry.items()
        }
        self.assertEqual(normalized, snapshot)

    def test_registry_modules_exist(self):
        registry = parse_registry()
        missing = [
            network_id
            for network_id, config in registry.items()
            if not (ROOT / "networks" / f"{config['network']}.py").is_file()
        ]
        self.assertEqual(missing, [])


class SavedAssetContractTests(unittest.TestCase):
    EXAMPLES = {
        "examples/general/assets/models/GCN_model_Aqsoldb_Solubility_20240101_000000": {
            "config.yaml", "final.pth", "learning_curve.csv", "learning_curve.png",
            "metrics.csv", "model_summary.txt", "network.py", "prediction.csv",
            "prediction.png", "scaler.pkl",
        },
        "examples/ISA/assets/Models/ISAT_model_Aqsoldb_Solubility_620_20240101_000000": {
            "config.yaml", "final.pth", "functional_group.csv",
            "learning_curve.csv", "learning_curve.png", "metrics.csv", "model_summary.txt",
            "network.py", "prediction.csv", "prediction.png", "scaler.pkl",
        },
        "examples/ISA/assets/Models/ISATPM_model_Aqsoldb_Solubility_620_20240101_000000": {
            "config.yaml", "final.pth", "functional_group.csv",
            "learning_curve.csv", "learning_curve.png", "metrics.csv", "model_summary.txt",
            "network.py", "prediction.csv", "prediction.png", "scaler.pkl",
        },
    }

    def test_example_model_composition(self):
        for relative_path, expected in self.EXAMPLES.items():
            with self.subTest(model=relative_path):
                actual = {
                    entry.name
                    for entry in (ROOT / relative_path).iterdir()
                    if entry.name not in {".ipynb_checkpoints", "data"}
                }
                self.assertEqual(actual, expected)

    def test_example_config_schema(self):
        common_keys = {
            "DATA_PATH", "FRAG_REF", "GRAPH_DIR", "MODEL_DIR", "MODEL_PATH", "NET_DIR",
            "NET_REFER", "batch_size", "conv_layers", "data", "data_manager_class",
            "data_manager_module", "description", "device", "dropout",
            "early_stopping_patience", "edge_dim", "hidden_dim", "learning_rate",
            "linear_layers", "lr_patience", "max_epoch", "min_lr", "name", "network",
            "network_manager_class", "network_manager_module", "node_dim", "optimizer",
            "pin_memory", "scaler", "solv_conv_layers", "solv_hidden_dim",
            "solv_linear_layers", "target", "target_dim", "train_manager_class",
            "train_manager_module", "version", "weight_decay",
        }
        for relative_path in self.EXAMPLES:
            with self.subTest(model=relative_path):
                config_keys = yaml_top_level_keys(ROOT / relative_path / "config.yaml")
                expected = set(common_keys)
                if "/ISA/" in relative_path:
                    expected.add("sculptor_index")
                self.assertEqual(config_keys, expected)


if __name__ == "__main__":
    unittest.main()
