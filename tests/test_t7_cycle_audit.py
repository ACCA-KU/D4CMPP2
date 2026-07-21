import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CycleAuditRegressionTests(unittest.TestCase):
    def test_public_batch_size_documentation_matches_runtime_default(self):
        source = (ROOT / "src" / "api" / "training.py").read_text(encoding="utf-8")
        module = ast.parse(source)
        config_assignment = next(
            node
            for node in module.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "config0"
                for target in node.targets
            )
        )
        config = ast.literal_eval(config_assignment.value)
        train_function = next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "train"
        )

        self.assertEqual(config["batch_size"], 256)
        self.assertIn("batch_size  : int. Default= 256", ast.get_docstring(train_function))

    def test_product_sources_do_not_use_bare_except(self):
        offenders = []
        for base in (ROOT / "src", ROOT / "networks"):
            for path in base.rglob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                offenders.extend(
                    f"{path.relative_to(ROOT)}:{node.lineno}"
                    for node in ast.walk(tree)
                    if isinstance(node, ast.ExceptHandler) and node.type is None
                )
        self.assertEqual(offenders, [])

    def test_matplotlib_rendering_avoids_removed_tostring_rgb(self):
        source = (ROOT / "src" / "utils" / "tools.py").read_text(encoding="utf-8")
        self.assertNotIn("tostring_rgb", source)
        self.assertIn("buffer_rgba", source)

    def test_release_readme_lists_current_network_ids_without_removed_docs_links(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

        self.assertIn(
            "| General | `GCN`, `MPNN`, `DMPNN`, `AFP`, `GAT` |",
            readme,
        )
        self.assertIn(
            "| With solvent | `GCNwS`, `MPNNwS`, `DMPNNwS`, `AFPwS`, `GATwS` |",
            readme,
        )
        self.assertIn("| ISA | `GC`, `ISAT`, `ISATPN` |", readme)
        self.assertNotIn("docs/", readme + changelog)


if __name__ == "__main__":
    unittest.main()
