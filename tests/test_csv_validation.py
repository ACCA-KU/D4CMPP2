import unittest

try:
    from .fixtures import load_source_module
except ImportError:
    from fixtures import load_source_module


validation = load_source_module("src/utils/csv_validation.py", "csv_validation_for_tests")


class CsvSchemaValidationTests(unittest.TestCase):
    def test_empty_csv_and_missing_columns_include_path_and_available_columns(self):
        with self.assertRaisesRegex(ValueError, r"empty\.csv.*no data rows"):
            validation.validate_csv_schema("empty.csv", ["compound", "target"], 0, ["compound"], ["target"], [])

        with self.assertRaisesRegex(ValueError, r"data\.csv.*molecule.*solvent.*numeric input.*temperature.*Available columns"):
            validation.validate_csv_schema(
                "data.csv", ["compound", "target"], 3,
                ["compound", "solvent"], ["target"], ["temperature"],
            )

    def test_duplicate_configured_columns_are_rejected(self):
        with self.assertRaisesRegex(ValueError, r"Duplicate.*target.*List each"):
            validation.validate_csv_schema(
                "data.csv", ["compound", "target"], 3,
                ["compound"], ["target", "target"], [],
            )

    def test_non_numeric_rows_and_all_nan_targets_are_actionable(self):
        with self.assertRaisesRegex(ValueError, r"Target.*target.*\[3, 7\].*numbers"):
            validation.validate_numeric_columns("data.csv", {"target": [3, 7]}, "Target")
        with self.assertRaisesRegex(ValueError, r"target_b.*only NaN.*at least one"):
            validation.validate_nonempty_targets("data.csv", {"target_a": False, "target_b": True})

    def test_csv_derived_array_lengths_must_match(self):
        validation.validate_aligned_lengths("data.csv", {"compound": 4, "target": 4, "numeric:x": 4})
        with self.assertRaisesRegex(ValueError, r"inconsistent lengths.*compound.*4.*target.*3"):
            validation.validate_aligned_lengths("data.csv", {"compound": 4, "target": 3})


class SplitValidationTests(unittest.TestCase):
    def test_allowed_explicit_labels_return_counts_and_test_is_optional(self):
        counts = validation.validate_set_labels("data.csv", ["train", "train", "val"])
        self.assertEqual(counts, {"train": 2, "val": 1, "test": 0})

    def test_invalid_and_missing_explicit_labels_show_counts(self):
        with self.assertRaisesRegex(ValueError, r"invalid labels.*'validation'.*Allowed labels.*Split counts"):
            validation.validate_set_labels("data.csv", ["train", "validation", "val"])
        with self.assertRaisesRegex(ValueError, r"missing required.*val.*train.*2.*val.*0"):
            validation.validate_set_labels("data.csv", ["train", "train"])

    def test_automatic_split_reports_actual_and_required_size(self):
        validation.validate_automatic_split_size("data.csv", 10)
        with self.assertRaisesRegex(ValueError, r"small\.csv.*9 usable rows.*at least 10.*'set'"):
            validation.validate_automatic_split_size("small.csv", 9)


if __name__ == "__main__":
    unittest.main()
