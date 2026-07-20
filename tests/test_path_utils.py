import os
from pathlib import Path
import shutil
import unittest

try:
    from .fixtures import TINY_REGRESSION_CSV, isolated_workdir, load_source_module
except ImportError:
    from fixtures import TINY_REGRESSION_CSV, isolated_workdir, load_source_module


class PathUtilityTests(unittest.TestCase):
    def load_path_module(self):
        return load_source_module("src/utils/PATH.py", f"d4cmpp2_path_test_{id(self)}")

    def test_init_path_uses_explicit_data_and_temp_runtime_directories(self):
        with isolated_workdir() as temporary:
            data_file = temporary / "tiny.csv"
            shutil.copyfile(TINY_REGRESSION_CSV, data_file)
            path_module = self.load_path_module()
            config = {"data": str(data_file.with_suffix(""))}

            path_module.init_path(config)

            self.assertEqual(Path(config["DATA_PATH"]), data_file)
            self.assertEqual(Path(config["MODEL_DIR"]), Path("_Models"))
            self.assertEqual(Path(config["GRAPH_DIR"]), Path("_Graphs"))
            self.assertTrue((temporary / "_Models").is_dir())
            self.assertTrue((temporary / "_Graphs").is_dir())

    def test_recursive_data_lookup_is_confined_to_workdir(self):
        with isolated_workdir() as temporary:
            nested = temporary / "nested" / "data"
            nested.mkdir(parents=True)
            data_file = nested / "tiny.csv"
            shutil.copyfile(TINY_REGRESSION_CSV, data_file)
            path_module = self.load_path_module()
            config = {"data": "tiny"}

            path_module.init_data_path(config)

            self.assertEqual(Path(config["DATA_PATH"]).resolve(), data_file.resolve())

    def test_explicit_runtime_paths_accept_platform_native_path_objects(self):
        with isolated_workdir() as temporary:
            model_dir = temporary / "runtime" / "models"
            graph_dir = temporary / "runtime" / "graphs"
            model_dir.mkdir(parents=True)
            graph_dir.mkdir(parents=True)
            path_module = self.load_path_module()
            config = {
                "data": "unused",
                "DATA_PATH": str(TINY_REGRESSION_CSV),
                "MODEL_DIR": str(model_dir),
                "GRAPH_DIR": str(graph_dir),
            }

            path_module.init_path(config)
            path_module.check_path(config)

            self.assertEqual(Path(config["MODEL_DIR"]), model_dir)
            self.assertEqual(Path(config["GRAPH_DIR"]), graph_dir)

    def test_model_path_is_created_below_model_dir(self):
        with isolated_workdir() as temporary:
            model_dir = temporary / "models"
            model_dir.mkdir()
            path_module = self.load_path_module()
            config = {
                "MODEL_DIR": str(model_dir),
                "network": "GCN",
                "data": "tiny",
                "target": ["target_a"],
            }

            result = Path(path_module.get_model_path(config))

            self.assertTrue(result.is_dir())
            self.assertEqual(result.parent, model_dir)
            self.assertTrue(result.name.startswith("GCN_tiny_target_a_"))

    def test_short_transfer_folder_name_is_preserved_in_output_name(self):
        with isolated_workdir() as temporary:
            model_dir = temporary / "models"
            model_dir.mkdir()
            path_module = self.load_path_module()
            config = {
                "MODEL_DIR": str(model_dir),
                "TRANSFER_PATH": str(temporary / "source"),
                "network": "GCN",
                "data": "tiny",
                "target": ["target_a"],
            }

            result = Path(path_module.get_model_path(config))

            self.assertTrue(result.name.startswith("source~GCN_tiny_target_a_"))

    def test_data_path_uses_only_file_stem_in_model_folder(self):
        with isolated_workdir() as temporary:
            model_dir = temporary / "models"
            model_dir.mkdir()
            data_path = temporary / "nested" / "tiny.csv"
            data_path.parent.mkdir()
            path_module = self.load_path_module()
            config = {
                "MODEL_DIR": str(model_dir),
                "network": "GCN",
                "data": str(data_path),
                "target": ["target_a"],
            }

            result = Path(path_module.get_model_path(config))

            self.assertEqual(result.parent, model_dir)
            self.assertTrue(result.name.startswith("GCN_tiny_target_a_"))

    def test_missing_data_and_model_errors_show_input_and_search_root(self):
        with isolated_workdir() as temporary:
            path_module = self.load_path_module()
            with self.assertRaisesRegex(FileNotFoundError, r"missing\.csv.*Searched.*DATA_PATH"):
                path_module.init_data_path({"data": "missing"})
            with self.assertRaisesRegex(FileNotFoundError, rf"missing-model.*{temporary.name}.*exact model path"):
                path_module.find_model_path("missing-model")

    def test_exact_model_path_wins_and_missing_path_like_value_fails_exactly(self):
        with isolated_workdir() as temporary:
            model = temporary / "outside" / "model"
            model.mkdir(parents=True)
            path_module = self.load_path_module()

            self.assertEqual(Path(path_module.find_model_path(model)), model)
            with self.assertRaisesRegex(FileNotFoundError, r"path-like.*not an existing directory.*not reinterpreted"):
                path_module.find_model_path(temporary / "outside" / "missing")

    def test_duplicate_model_and_data_basenames_are_ambiguous(self):
        with isolated_workdir() as temporary:
            path_module = self.load_path_module()
            for parent in ("first", "second"):
                (temporary / parent / "same-model").mkdir(parents=True)
                (temporary / parent / "same.csv").write_text("value\n1\n", encoding="utf-8")

            with self.assertRaisesRegex(FileNotFoundError, r"same-model.*ambiguous.*Candidates.*exact model path"):
                path_module.find_model_path("same-model")
            with self.assertRaisesRegex(FileNotFoundError, r"same\.csv.*ambiguous.*Candidates.*exact path"):
                path_module.init_data_path({"data": "same"})

    def test_explicit_output_model_path_is_honored(self):
        with isolated_workdir() as temporary:
            model_dir = temporary / "models"
            model_dir.mkdir()
            explicit = temporary / "custom" / "run"
            path_module = self.load_path_module()
            config = {"MODEL_DIR": str(model_dir), "MODEL_PATH": str(explicit)}

            result = Path(path_module.get_model_path(config))

            self.assertEqual(result, explicit)
            self.assertTrue(explicit.is_dir())

    def test_data_and_runtime_defaults_follow_call_time_working_directory(self):
        with isolated_workdir() as temporary:
            path_module = self.load_path_module()
            later = temporary / "later"
            later.mkdir()
            data = later / "later.csv"
            data.write_text("value\n1\n", encoding="utf-8")
            os.chdir(later)

            config = {"data": "later"}
            self.assertEqual(Path(path_module.init_data_path(config)).resolve(), data.resolve())
            self.assertEqual(Path(path_module.get_xyz_dir_path({})).resolve(), (later / "_XYZ").resolve())

    def test_explicit_runtime_path_errors_identify_key_value_and_fix(self):
        with isolated_workdir() as temporary:
            path_module = self.load_path_module()
            missing = temporary / "missing"
            cases = (
                ({"DATA_PATH": str(missing), "GRAPH_DIR": str(temporary), "MODEL_DIR": str(temporary)}, "DATA_PATH", "existing file"),
                ({"DATA_PATH": str(TINY_REGRESSION_CSV), "GRAPH_DIR": str(missing), "MODEL_DIR": str(temporary)}, "GRAPH_DIR", "Create it"),
                ({"DATA_PATH": str(TINY_REGRESSION_CSV), "GRAPH_DIR": str(temporary), "MODEL_DIR": str(missing)}, "MODEL_DIR", "Create it"),
            )
            for config, key, hint in cases:
                with self.subTest(key=key), self.assertRaisesRegex(FileNotFoundError, rf"{key}.*missing.*{hint}"):
                    path_module.check_path(config)

    def test_support_path_errors_include_explicit_value_and_expected_type(self):
        with isolated_workdir() as temporary:
            path_module = self.load_path_module()
            missing = temporary / "missing"
            with self.assertRaisesRegex(FileNotFoundError, r"NET_REFER.*missing.*YAML"):
                path_module.get_network_refer({"NET_REFER": str(missing)})
            with self.assertRaisesRegex(FileNotFoundError, r"FRAG_REF.*missing.*CSV"):
                path_module.get_frag_ref_path({"FRAG_REF": str(missing)})
            with self.assertRaisesRegex(FileNotFoundError, r"NET_DIR.*missing.*existing directory"):
                path_module.get_NET_DIR({"NET_DIR": str(missing)})
            with self.assertRaisesRegex(FileNotFoundError, r"xyz_dir.*missing.*XYZ directory"):
                path_module.get_xyz_dir_path({"xyz_dir": str(missing)})


class WorkspaceIsolationTests(unittest.TestCase):
    def test_fast_suite_does_not_create_runtime_directories_in_repository(self):
        repository = Path(__file__).resolve().parents[1]
        unexpected = [name for name in ("_Models", "_Graphs", "_XYZ") if (repository / name).exists()]
        self.assertEqual(unexpected, [])


if __name__ == "__main__":
    unittest.main()
