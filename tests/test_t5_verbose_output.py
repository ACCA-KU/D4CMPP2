import io
from pathlib import Path
import warnings
import unittest

from fixtures import ROOT, load_source_module
from markers import heavy_test


output = load_source_module("src/utils/output.py", "t5_verbose_output")


class VerboseOutputTests(unittest.TestCase):
    def test_default_and_verbose_true_show_information(self):
        for config in (None, {}, {"verbose": True}):
            with self.subTest(config=config):
                stream = io.StringIO()
                adapter = output.OutputAdapter(
                    verbose=True if config is None else config.get("verbose", True),
                    stream=stream,
                )
                adapter.info("[Training] Starting a new run.")
                self.assertIn("[Training]", stream.getvalue())

    def test_verbose_false_hides_info_but_not_error_or_warning(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        adapter = output.OutputAdapter(
            verbose=False,
            stream=stdout,
            error_stream=stderr,
        )
        adapter.info("hidden")
        adapter.error("[Error] visible")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            adapter.warning("[Data] fallback visible")

        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("[Error] visible", stderr.getvalue())
        self.assertEqual(len(caught), 1)
        self.assertIn("fallback visible", str(caught[0].message))

    @heavy_test
    def test_verbose_false_disables_progress(self):
        adapter = output.OutputAdapter(verbose=False)
        progress = adapter.progress([1, 2, 3])
        try:
            self.assertTrue(progress.disable)
            self.assertEqual(list(progress), [1, 2, 3])
        finally:
            progress.close()

    def test_product_prints_are_centralized(self):
        allowed = {
            ROOT / "src" / "utils" / "output.py",
        }
        offenders = []
        paths = list(ROOT.glob("*.py"))
        paths.extend((ROOT / "src").rglob("*.py"))
        paths.extend((ROOT / "networks").rglob("*.py"))
        for path in paths:
            relative = path.relative_to(ROOT)
            if any(part.startswith(".") for part in relative.parts):
                continue
            if path in allowed:
                continue
            for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if "print(" in stripped:
                    offenders.append(f"{relative}:{number}")
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
