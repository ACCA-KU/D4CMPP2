import unittest

try:
    from .fixtures import load_source_module
except ImportError:
    from fixtures import load_source_module


validation = load_source_module("src/utils/config_validation.py", "config_validation_for_tests")


def valid_config(**overrides):
    config = {
        "network": "GCN",
        "data_manager_module": "MolDataManager",
        "scaler": "standard",
        "optimizer": "Adam",
        "max_epoch": 2,
        "batch_size": 4,
        "learning_rate": 0.001,
        "weight_decay": 0.0005,
        "lr_patience": 2,
        "early_stopping_patience": 2,
        "min_lr": 1e-5,
        "device": "cpu",
        "pin_memory": False,
        "legacy_silent_errors": False,
    }
    config.update(overrides)
    return config


class EntryArgumentValidationTests(unittest.TestCase):
    def test_training_path_modes_are_mutually_exclusive(self):
        module = load_source_module("src/utils/config_validation.py", "config_validation_resume_modes")
        with self.assertRaisesRegex(
            ValueError,
            "mutually exclusive.*LOAD_PATH.*RESUME_PATH",
        ):
            module.validate_entry_args(
                {"LOAD_PATH": "legacy", "RESUME_PATH": "exact"}
            )
        module.validate_entry_args({"RESUME_PATH": "model"})

    def test_valid_scratch_args_are_not_mutated(self):
        args = {"data": "tiny.csv", "target": ["target"], "network": "GCN"}
        original = dict(args)
        validation.validate_entry_args(args)
        self.assertEqual(args, original)

    def test_target_requires_non_empty_string_list_with_example(self):
        for target in ("target", [], [""], [1]):
            with self.subTest(target=target), self.assertRaisesRegex(TypeError, r"target .*Example"):
                validation.validate_entry_args({"data": "tiny", "target": target, "network": "GCN"})

    def test_data_and_network_are_required_strings(self):
        with self.assertRaisesRegex(TypeError, r"data .*Aqsoldb\.csv"):
            validation.validate_entry_args({"target": ["target"], "network": "GCN"})
        with self.assertRaisesRegex(TypeError, r"network .*network='GCN'"):
            validation.validate_entry_args({"data": "tiny", "target": ["target"]})


class RegistryValidationTests(unittest.TestCase):
    ENTRY = {field: field for field in validation.REGISTRY_FIELDS}

    def test_unknown_network_lists_ids_and_close_match(self):
        with self.assertRaisesRegex(ValueError, r"GCNN.*GCN.*Did you mean"):
            validation.validate_network_entry("GCNN", None, ["GCN", "GAT"])

    def test_missing_manager_key_is_reported_before_loading(self):
        entry = dict(self.ENTRY)
        del entry["train_manager_class"]
        with self.assertRaisesRegex(ValueError, r"GCN.*train_manager_class.*YAML"):
            validation.validate_network_entry("GCN", entry, ["GCN"])


class MergedConfigValidationTests(unittest.TestCase):
    def test_normal_config_is_not_mutated(self):
        config = valid_config()
        original = dict(config)
        validation.validate_training_config(config, optimizer_names={"Adam", "SGD"})
        self.assertEqual(config, original)

    def test_scaler_and_optimizer_are_actionable(self):
        with self.assertRaisesRegex(ValueError, r"scaler.*standard.*unknown"):
            validation.validate_training_config(valid_config(scaler="unknown"))
        with self.assertRaisesRegex(ValueError, r"Adma.*Did you mean 'Adam'"):
            validation.validate_training_config(valid_config(optimizer="Adma"), optimizer_names={"Adam", "SGD"})

    def test_numeric_ranges_reject_bool_zero_negative_and_min_lr_mismatch(self):
        cases = (
            ("max_epoch", 0),
            ("batch_size", True),
            ("learning_rate", 0),
            ("weight_decay", -1),
            ("lr_patience", -1),
            ("early_stopping_patience", 1.5),
        )
        for key, value in cases:
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                validation.validate_training_config(valid_config(**{key: value}))
        with self.assertRaisesRegex(ValueError, r"min_lr.*learning_rate"):
            validation.validate_training_config(valid_config(learning_rate=1e-5, min_lr=1e-4))

    def test_device_and_pin_memory_types_are_checked(self):
        for device in (0, "gpu", "cuda:-1", "cuda:one"):
            with self.subTest(device=device), self.assertRaisesRegex(ValueError, r"device.*cuda:0"):
                validation.validate_training_config(valid_config(device=device))
        with self.assertRaisesRegex(TypeError, r"pin_memory.*bool"):
            validation.validate_training_config(valid_config(pin_memory="false"))
        with self.assertRaisesRegex(TypeError, r"legacy_silent_errors.*bool.*legacy compatibility"):
            validation.validate_training_config(valid_config(legacy_silent_errors="false"))
        with self.assertRaisesRegex(TypeError, r"verbose.*bool.*verbose=False"):
            validation.validate_training_config(valid_config(verbose="false"))

    def test_target_scaler_fit_scope_is_explicit(self):
        validation.validate_training_config(valid_config(target_scaler_fit_scope="train"))
        validation.validate_training_config(valid_config(target_scaler_fit_scope="all"))
        with self.assertRaisesRegex(ValueError, r"target_scaler_fit_scope.*train.*all.*legacy"):
            validation.validate_training_config(valid_config(target_scaler_fit_scope="validation"))

    def test_isa_sculptor_errors_distinguish_list_missing_and_invalid_values(self):
        base = valid_config(network="ISAT", data_manager_module="ISADataManager")
        with self.assertRaisesRegex(TypeError, r"sculptor_index.*tuple.*list"):
            validation.validate_training_config({**base, "sculptor_index": [6, 2, 0]})
        with self.assertRaisesRegex(ValueError, r"requires sculptor_index.*sculptor_s"):
            validation.validate_training_config(base)
        with self.assertRaisesRegex(ValueError, r"non-negative integers.*sculptor_c"):
            validation.validate_training_config({**base, "sculptor_s": 6, "sculptor_c": -1, "sculptor_a": 0})

    def test_legacy_sculptor_tuple_is_checked_before_normalization(self):
        validation.validate_sculptor_index_argument((6, 2, 0))
        for value in ((6, 2), (6, 2, -1), (6, True, 0)):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, r"three non-negative integers"):
                validation.validate_sculptor_index_argument(value)

    def test_graph_cache_policy_is_explicit_and_validated(self):
        base = valid_config()
        for policy in ("v2", "legacy", "regenerate"):
            validation.validate_training_config({**base, "graph_cache_policy": policy})
        with self.assertRaisesRegex(ValueError, "graph_cache_policy.*v2.*legacy.*regenerate"):
            validation.validate_training_config({**base, "graph_cache_policy": "auto"})

        validation.validate_training_config({**base, "validate_graph_cache": False})
        with self.assertRaisesRegex(TypeError, "validate_graph_cache.*bool"):
            validation.validate_training_config({**base, "validate_graph_cache": "false"})
        validation.validate_training_config({**base, "data_quality_report": False})
        with self.assertRaisesRegex(TypeError, "data_quality_report.*bool"):
            validation.validate_training_config({**base, "data_quality_report": "false"})

    def test_reproducibility_policy_requires_valid_seed_and_bool(self):
        base = valid_config()
        validation.validate_training_config({**base, "random_seed": 0})
        validation.validate_training_config({
            **base,
            "random_seed": 42,
            "deterministic_algorithms": True,
        })
        for seed in (True, -1, 1.5, "42"):
            with self.subTest(seed=seed), self.assertRaisesRegex(ValueError, "random_seed"):
                validation.validate_training_config({**base, "random_seed": seed})
        with self.assertRaisesRegex(TypeError, "deterministic_algorithms.*bool"):
            validation.validate_training_config({
                **base,
                "deterministic_algorithms": "true",
            })
        with self.assertRaisesRegex(ValueError, "requires random_seed"):
            validation.validate_training_config({
                **base,
                "deterministic_algorithms": True,
            })


class ValidationWiringTests(unittest.TestCase):
    def test_training_entry_point_calls_all_early_validation_boundaries(self):
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[1] / "src" / "api" / "training.py"
        ).read_text(encoding="utf-8")
        for call in (
            "validate_entry_args(kwargs)",
            "validate_network_entry(config['network'], net_config",
            "validate_sculptor_index_argument(config['sculptor_index'])",
            "validate_training_config(config, optimizer_names=optimizer_names)",
            'validate_runtime_environment(config, backend="pyg", torch_module=torch)',
        ):
            with self.subTest(call=call):
                self.assertIn(call, source)


if __name__ == "__main__":
    unittest.main()
