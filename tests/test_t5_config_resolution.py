import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fixtures import load_source_module
from markers import heavy_test


resolution_module = load_source_module(
    "src/utils/config_resolution.py",
    "t5_config_resolution_contract",
)


class ConfigResolutionUnitTests(unittest.TestCase):
    def test_layers_are_deep_copied_and_unknown_keys_are_preserved(self):
        source = {"nested": {"values": [1]}, "custom_network_key": "kept"}
        values, provenance = resolution_module.merge_config_layers(
            (
                ("defaults", {"learning_rate": 0.001, "nested": {"old": True}}),
                ("custom", source),
            )
        )
        source["nested"]["values"].append(2)

        self.assertEqual(values["nested"], {"values": [1]})
        self.assertEqual(values["custom_network_key"], "kept")
        self.assertEqual(provenance["learning_rate"], "defaults")
        self.assertEqual(provenance["custom_network_key"], "custom")

        resolution = resolution_module.ConfigResolution.from_working(
            values,
            provenance,
        )
        with self.assertRaises(TypeError):
            resolution.values["new"] = "value"
        first = resolution.to_dict()
        first["nested"]["values"].append(3)
        self.assertEqual(resolution.values["nested"], {"values": [1]})

    def test_invalid_layer_reports_source(self):
        with self.assertRaisesRegex(TypeError, "broken.*mapping.*list"):
            resolution_module.merge_config_layers((("broken", []),))

    def test_runtime_split_does_not_mutate_pipeline_input(self):
        config = {
            "MODEL_PATH": "model",
            "TRANSFER_PATH": "source",
            "loaded": False,
            "custom": {"items": [1]},
        }
        working, runtime = resolution_module.split_runtime_config(config)
        working["custom"]["items"].append(2)

        self.assertEqual(
            config,
            {
                "MODEL_PATH": "model",
                "TRANSFER_PATH": "source",
                "loaded": False,
                "custom": {"items": [1]},
            },
        )
        self.assertNotIn("TRANSFER_PATH", working)
        self.assertEqual(runtime.transfer_path, "source")
        self.assertFalse(runtime.loaded)

    def test_overlay_preserves_untouched_provenance(self):
        values, provenance = resolution_module.merge_config_layers(
            (
                ("defaults", {"learning_rate": 0.001}),
                ("api", {"custom": 4}),
            )
        )
        values, provenance = resolution_module.overlay_config_layer(
            values,
            provenance,
            {"learning_rate": 0.02},
            source="registry",
        )
        self.assertEqual(provenance["custom"], "api")
        self.assertEqual(provenance["learning_rate"], "registry")


class ConfigResolutionIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys

        from fixtures import ROOT

        if str(ROOT.parent) not in sys.path:
            sys.path.insert(0, str(ROOT.parent))

    def _patch_boundaries(self, main, model_path):
        registry = {
            "network": "GCN_model",
            "data_manager_module": "MolDataManager",
            "data_manager_class": "MolDataManager",
            "network_manager_module": "NetworkManager",
            "network_manager_class": "NetworkManager",
            "train_manager_module": "TrainManager",
            "train_manager_class": "Trainer",
            "learning_rate": 0.02,
        }
        return (
            mock.patch.object(main, "load_NET_REFER", return_value=registry),
            mock.patch.object(main.PATH, "init_path"),
            mock.patch.object(main.PATH, "check_path"),
            mock.patch.object(main.PATH, "get_model_path", return_value=str(model_path)),
            mock.patch.object(main, "validate_runtime_environment", return_value={}),
        )

    @heavy_test
    def test_scratch_registry_precedence_and_input_nonmutation_are_preserved(self):
        from D4CMPP2 import _main

        with tempfile.TemporaryDirectory(prefix="d4cmpp2-config-scratch-") as temporary:
            kwargs = {
                "data": "tiny",
                "target": ["target"],
                "target_dim": 1,
                "network": "GCN",
                "device": "cpu",
                "learning_rate": 0.5,
                "custom_network_key": {"nested": [1]},
            }
            original = {
                **kwargs,
                "target": list(kwargs["target"]),
                "custom_network_key": {"nested": [1]},
            }
            patches = self._patch_boundaries(_main, Path(temporary) / "model")
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                resolution = _main.resolve_config(**kwargs)

            self.assertEqual(kwargs, original)
            self.assertEqual(resolution.values["learning_rate"], 0.02)
            self.assertEqual(resolution.provenance["learning_rate"], "registry")
            self.assertEqual(
                resolution.values["custom_network_key"],
                {"nested": [1]},
            )
            self.assertEqual(
                resolution.provenance["custom_network_key"],
                "api_or_cli",
            )

    @heavy_test
    def test_legacy_load_default_overwrite_and_saved_unknown_key_are_preserved(self):
        import yaml

        from D4CMPP2 import _main

        with tempfile.TemporaryDirectory(prefix="d4cmpp2-config-load-") as temporary:
            model_path = Path(temporary)
            saved = {
                "network": "GCN_model",
                "data": "tiny",
                "target": ["target"],
                "target_dim": 1,
                "data_manager_module": "MolDataManager",
                "data_manager_class": "MolDataManager",
                "network_manager_module": "NetworkManager",
                "network_manager_class": "NetworkManager",
                "train_manager_module": "TrainManager",
                "train_manager_class": "Trainer",
                "scaler": "standard",
                "optimizer": "Adam",
                "max_epoch": 7,
                "batch_size": 99,
                "learning_rate": 0.03,
                "weight_decay": 0.0,
                "lr_patience": 2,
                "early_stopping_patience": 3,
                "min_lr": 1e-5,
                "device": "cpu",
                "pin_memory": False,
                "custom_saved_key": {"value": 4},
            }
            (model_path / "config.yaml").write_text(
                yaml.safe_dump(saved),
                encoding="utf-8",
            )
            with (
                mock.patch.object(
                    _main.PATH,
                    "find_model_path",
                    return_value=str(model_path),
                ),
                mock.patch.object(_main.PATH, "init_path"),
                mock.patch.object(_main.PATH, "check_path"),
                mock.patch.object(
                    _main,
                    "validate_runtime_environment",
                    return_value={},
                ),
            ):
                resolution = _main.resolve_config(LOAD_PATH="saved")

            self.assertEqual(resolution.values["batch_size"], 256)
            self.assertEqual(
                resolution.provenance["batch_size"],
                "legacy_load_defaults_or_override",
            )
            self.assertEqual(
                resolution.values["custom_saved_key"],
                {"value": 4},
            )
            self.assertEqual(
                resolution.provenance["custom_saved_key"],
                "saved_load",
            )


if __name__ == "__main__":
    unittest.main()
