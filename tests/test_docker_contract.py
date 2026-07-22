from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
DOCKER_DIR = ROOT / "docker"


class DockerContractTests(unittest.TestCase):
    def test_expected_variants_are_present(self):
        self.assertEqual(
            {path.name for path in DOCKER_DIR.glob("Dockerfile.*")},
            {"Dockerfile.cpu", "Dockerfile.cuda128", "Dockerfile.cuda130"},
        )

    def test_images_build_a_wheel_and_run_as_non_root(self):
        for name in ("cpu", "cuda128", "cuda130"):
            with self.subTest(name=name):
                dockerfile = (DOCKER_DIR / f"Dockerfile.{name}").read_text()
                self.assertIn("AS wheel-builder", dockerfile)
                self.assertIn("python -m build --wheel", dockerfile)
                self.assertIn("COPY --from=wheel-builder", dockerfile)
                self.assertIn("import D4CMPP2, torch, torch_geometric", dockerfile)
                self.assertIn("USER d4cmpp2", dockerfile)

    def test_cuda_names_match_the_actual_pytorch_runtime(self):
        cuda128 = (DOCKER_DIR / "Dockerfile.cuda128").read_text()
        cuda130 = (DOCKER_DIR / "Dockerfile.cuda130").read_text()
        self.assertIn("cuda12.8-cudnn9-runtime", cuda128)
        self.assertIn("torch.version.cuda == '12.8'", cuda128)
        self.assertIn("cuda13.0-cudnn9-runtime", cuda130)
        self.assertIn("torch.version.cuda == '13.0'", cuda130)

    def test_documented_commands_cover_all_variants(self):
        readme = (DOCKER_DIR / "README.md").read_text()
        for name in ("cpu", "cuda128", "cuda130"):
            self.assertIn(f"docker/Dockerfile.{name}", readme)
            self.assertIn(f"1.0.0-{name}", readme)


if __name__ == "__main__":
    unittest.main()
