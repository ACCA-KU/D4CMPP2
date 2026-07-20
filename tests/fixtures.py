import contextlib
import importlib.util
import os
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent / "data"
TINY_REGRESSION_CSV = DATA_DIR / "tiny_regression.csv"


@contextlib.contextmanager
def isolated_workdir():
    """Run code in a temporary cwd and always restore the caller's cwd."""
    original = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="d4cmpp2-test-") as temporary:
        temporary_path = Path(temporary)
        os.chdir(temporary_path)
        try:
            yield temporary_path
        finally:
            os.chdir(original)


def load_source_module(relative_path, module_name):
    """Load one source file without importing the dependency-heavy package."""
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

