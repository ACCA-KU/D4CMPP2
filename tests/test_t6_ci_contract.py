from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


class CIContractTests(unittest.TestCase):
    def test_release_workflow_uses_verified_artifact_and_job_scoped_oidc(self):
        release = (ROOT / ".github" / "workflows" / "release.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("types: [published]", release)
        self.assertIn("python -m build --outdir dist", release)
        self.assertIn("check_distribution_artifacts.py dist", release)
        self.assertIn("actions/upload-artifact@v4", release)
        self.assertIn("actions/download-artifact@v4", release)
        self.assertIn("needs: build", release)
        self.assertIn("name: pypi", release)
        self.assertIn("id-token: write", release)
        self.assertIn("pypa/gh-action-pypi-publish@release/v1", release)
        self.assertIn("actions/checkout@v5", release)
        self.assertIn("actions/setup-python@v6", release)
        self.assertIn('tag_version="${GITHUB_REF_NAME#v}"', release)
        self.assertIn('python-version: "3.11"', release)
        self.assertIn("import tomllib", release)
        self.assertNotIn("PYPI_TOKEN", release)
        self.assertNotIn("password:", release)

    def test_ci_covers_declared_python_and_primary_operating_system_matrix(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        fast_unit = workflow.split("  fast-unit:", 1)[1].split("  pyg-cpu:", 1)[0]

        self.assertIn('python: ["3.10", "3.11", "3.12"]', workflow)
        self.assertIn("os: [ubuntu-latest, windows-latest]", workflow)
        self.assertIn("python -m unittest discover -s tests -v", workflow)
        self.assertIn("actions/checkout@v5", workflow)
        self.assertIn("actions/setup-python@v6", workflow)
        self.assertNotIn("cache: pip", fast_unit)

    def test_ci_has_full_pyg_cpu_and_clean_artifact_jobs(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("torch==2.2.0 torch-geometric==2.8.0", workflow)
        self.assertIn('python -m pip install "numpy<2"', workflow)
        self.assertIn("profile: latest", workflow)
        self.assertIn('python: "3.12"', workflow)
        self.assertIn("D4CMPP2_RUN_HEAVY: \"1\"", workflow)
        self.assertIn('CUDA_VISIBLE_DEVICES: ""', workflow)
        self.assertIn("bash tests/ci_artifact_checks.sh", workflow)

    def test_ci_pins_incremental_lint_and_type_checks(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        pyright_config = (ROOT / "pyrightconfig.json").read_text(encoding="utf-8")
        static_script = (ROOT / "tests" / "ci_static_checks.sh").read_text(encoding="utf-8")

        self.assertIn("bash tests/ci_static_checks.sh", workflow)
        self.assertIn("ruff==0.15.3 pyright==1.1.410", static_script)
        self.assertIn('ruff check --select F63,F7,F82 "${typed_modules[@]}"', static_script)
        self.assertIn("pyright --project pyrightconfig.json", static_script)
        self.assertIn('"typeCheckingMode": "basic"', pyright_config)
        self.assertIn('"src/DataManager/contracts.py"', pyright_config)

    def test_ci_uses_read_only_repository_permissions(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("permissions:\n  contents: read", workflow)

    def test_release_metadata_maintenance_guides_and_artifact_checks_are_present(self):
        artifact_script = (ROOT / "tests" / "ci_artifact_checks.sh").read_text(
            encoding="utf-8"
        )
        wheel_smoke = (ROOT / "tests" / "wheel_install_smoke.py").read_text(
            encoding="utf-8"
        )
        for path in (
            ROOT / "LICENSE",
            ROOT / "MANIFEST.in",
            ROOT / "README.md",
            ROOT / "CHANGELOG.md",
            ROOT / "AGENTS.md",
            ROOT / "networks" / "AGENTS.md",
            ROOT / "src" / "Analyzer" / "AGENTS.md",
            ROOT / "src" / "DataManager" / "AGENTS.md",
            ROOT / "src" / "NetworkManager" / "AGENTS.md",
            ROOT / "src" / "TrainManager" / "AGENTS.md",
            ROOT / "src" / "utils" / "AGENTS.md",
            ROOT / "tests" / "AGENTS.md",
            ROOT / "tests" / "check_distribution_artifacts.py",
        ):
            self.assertTrue(path.is_file(), path)
        self.assertEqual(list((ROOT / "docs").glob("*.md")), [])
        self.assertIn("python -m build", artifact_script)
        self.assertIn("check_distribution_artifacts.py", artifact_script)
        self.assertIn("wheel_install_smoke.py", artifact_script)
        self.assertIn('"--no-deps", "--force-reinstall", str(wheel)', wheel_smoke)

    def test_dependency_metadata_uses_approved_major_version_bounds(self):
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")

        for source in (pyproject, requirements):
            self.assertIn("torch>=2.2,<3", source)
            self.assertIn("torch-geometric>=2.8,<3", source)


if __name__ == "__main__":
    unittest.main()
