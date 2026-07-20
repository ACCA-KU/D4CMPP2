import contextlib
import io
import unittest

from fixtures import load_source_module


argparser = load_source_module("src/utils/argparser.py", "argparser_for_tests")


class CliArgumentTests(unittest.TestCase):
    def test_resume_path_is_parsed_without_scratch_arguments(self):
        args = argparser.parse_args(["--resume", "model", "--epoch", "3"])
        self.assertEqual(args.RESUME_PATH, "model")
        self.assertEqual(args.max_epoch, 3)

    def test_seed_and_deterministic_flags_are_parsed(self):
        args = argparser.parse_args(["--seed", "42", "--deterministic"])
        self.assertEqual(args.random_seed, 42)
        self.assertTrue(args.deterministic_algorithms)

    def test_quiet_disables_verbose_output(self):
        args = argparser.parse_args(["--quiet"])
        self.assertFalse(args.verbose)

    def test_cuda_zero_target_and_fragment_are_preserved(self):
        args = argparser.parse_args([
            "--data", "tiny.csv", "--target", "a, b", "--network", "GCN",
            "--cuda", "0", "--fragment", "6,2,0",
        ])
        self.assertEqual(args.device, "cuda:0")
        self.assertEqual(args.target, ["a", "b"])
        self.assertEqual(args.sculptor_index, (6, 2, 0))

    def test_load_does_not_discard_later_options(self):
        args = argparser.parse_args(["--load", "model", "--device", "cpu", "--epoch", "3"])
        self.assertEqual(args.LOAD_PATH, "model")
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.max_epoch, 3)

    def test_help_lists_cpu_usage_and_supported_ids(self):
        stream = io.StringIO()
        with self.assertRaises(SystemExit) as caught, contextlib.redirect_stdout(stream):
            argparser.parse_args(["--help"])
        self.assertEqual(caught.exception.code, 0)
        help_text = stream.getvalue()
        self.assertIn("--device", help_text)
        self.assertIn("cpu", help_text)
        self.assertIn("ISATPN", help_text)


if __name__ == "__main__":
    unittest.main()
