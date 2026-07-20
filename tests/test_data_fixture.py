import csv
import unittest

from fixtures import TINY_REGRESSION_CSV


class TinyDataFixtureTests(unittest.TestCase):
    def test_fixture_schema_and_size(self):
        with TINY_REGRESSION_CSV.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))

        self.assertEqual(list(rows[0]), ["compound", "target_a", "target_b", "set"])
        self.assertEqual(len(rows), 12)
        self.assertEqual({row["set"] for row in rows}, {"train", "val", "test"})
        self.assertTrue(any(row["target_b"] == "" for row in rows))
        self.assertTrue(all(row["compound"] for row in rows))

    def test_fixture_has_no_download_or_absolute_path_dependency(self):
        self.assertTrue(TINY_REGRESSION_CSV.is_file())
        text = TINY_REGRESSION_CSV.read_text(encoding="utf-8")
        self.assertNotIn("http://", text)
        self.assertNotIn("https://", text)
        self.assertNotIn(":\\", text)


if __name__ == "__main__":
    unittest.main()

