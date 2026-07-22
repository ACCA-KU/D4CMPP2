"""Build and install the wheel outside the source checkout for packaging QA."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def run(*command, cwd=None):
    subprocess.run(command, cwd=cwd, check=True)


def main():
    with tempfile.TemporaryDirectory(prefix="d4cmpp2-wheel-") as temporary:
        temporary = Path(temporary)
        editable_venv = temporary / "editable-venv"
        run(sys.executable, "-m", "venv", "--system-site-packages", str(editable_venv))
        editable_python = editable_venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        run(str(editable_python), "-m", "pip", "install", "--no-deps", "-e", str(ROOT))
        editable_probe = temporary / "editable-probe"
        editable_probe.mkdir()
        run(
            str(editable_python), "-I", "-c",
            "import importlib.metadata as m, D4CMPP2; "
            "assert D4CMPP2.__version__ == m.version('D4CMPP2'); "
            "print(D4CMPP2.__file__)",
            cwd=editable_probe,
        )
        dist = temporary / "dist"
        dist.mkdir()
        run(
            sys.executable,
            "-m",
            "pip",
            "wheel",
            str(ROOT),
            "--no-deps",
            "-w",
            str(dist),
        )
        wheel = next(dist.glob("*.whl"))
        venv = temporary / "venv"
        # Reuse the wrapper's already verified ML binaries while isolating the D4CMPP2 artifact.
        run(sys.executable, "-m", "venv", "--system-site-packages", str(venv))
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        # The CI parent environment already has D4CMPP2 installed so runtime
        # dependencies are available through --system-site-packages. Force pip
        # to install this wheel into the child venv instead of treating the
        # parent copy with the same version as already satisfied.
        run(
            str(python), "-m", "pip", "install",
            "--no-deps", "--force-reinstall", str(wheel),
        )
        probe = temporary / "probe"
        probe.mkdir()
        code = (
            "import importlib.metadata as m, importlib.resources as r, D4CMPP2; "
            "assert D4CMPP2.__version__ == m.version('D4CMPP2'); "
            "root=r.files('D4CMPP2'); "
            "assert root.joinpath('network_refer.yaml').is_file(); "
            "assert root.joinpath('src/utils/functional_group.csv').is_file(); "
            "assert root.joinpath('_Data/test.csv').is_file(); "
            "print(D4CMPP2.__version__, D4CMPP2.__file__)"
        )
        run(str(python), "-I", "-c", code, cwd=probe)
        run(str(python), "-I", "-m", "D4CMPP2", "--help", cwd=probe)
        executable = venv / ("Scripts/d4cmpp2.exe" if os.name == "nt" else "bin/d4cmpp2")
        run(
            str(executable), "--data", "test", "--target", "Abs", "--network", "GCN",
            "--device", "cpu", "--epoch", "1", cwd=probe,
        )


if __name__ == "__main__":
    main()
