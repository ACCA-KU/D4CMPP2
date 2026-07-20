"""Validate the contents and metadata of built sdist and wheel artifacts."""

from email.parser import BytesParser
from pathlib import Path
import sys
import tarfile
import zipfile


REQUIRED_PACKAGE_DATA = {
    "D4CMPP2/network_refer.yaml",
    "D4CMPP2/src/utils/functional_group.csv",
    "D4CMPP2/_Data/test.csv",
}


def _single(directory: Path, pattern: str) -> Path:
    matches = list(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {pattern!r} artifact in {directory}, found {matches!r}."
        )
    return matches[0]


def check_wheel(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        missing = REQUIRED_PACKAGE_DATA - names
        if missing:
            raise RuntimeError(f"Wheel {path.name!r} is missing package data: {sorted(missing)!r}.")
        metadata_name = next(
            (name for name in names if name.endswith(".dist-info/METADATA")),
            None,
        )
        if metadata_name is None:
            raise RuntimeError(f"Wheel {path.name!r} has no .dist-info/METADATA file.")
        license_names = [
            name
            for name in names
            if ".dist-info/" in name and name.lower().endswith("/license")
        ]
        if not license_names:
            raise RuntimeError(f"Wheel {path.name!r} does not include LICENSE.")
        metadata = BytesParser().parsebytes(archive.read(metadata_name))

    required = set(metadata.get_all("Requires-Dist", []))
    if not any(value.startswith("torch") and "<3" in value and ">=2.2" in value for value in required):
        raise RuntimeError("Wheel metadata does not contain the approved torch>=2.2,<3 bound.")
    if not any(
        value.startswith("torch-geometric") and "<3" in value and ">=2.8" in value
        for value in required
    ):
        raise RuntimeError(
            "Wheel metadata does not contain the approved torch-geometric>=2.8,<3 bound."
        )
    python_bounds = {
        item.strip() for item in (metadata["Requires-Python"] or "").split(",")
    }
    if python_bounds != {">=3.10", "<3.13"}:
        raise RuntimeError(
            f"Unexpected Requires-Python metadata: {metadata['Requires-Python']!r}."
        )


def check_sdist(path: Path) -> None:
    with tarfile.open(path, "r:gz") as archive:
        names = archive.getnames()
    relative = {name.split("/", 1)[1] for name in names if "/" in name}
    required = {
        "LICENSE",
        "MANIFEST.in",
        "README.md",
        "pyproject.toml",
        "network_refer.yaml",
        "src/utils/functional_group.csv",
        "_Data/test.csv",
        "examples/README.md",
        "examples/assets/tiny_numeric.csv",
        "examples/training/01_basic_cpu.py",
        "examples/inference/01_prediction.py",
        "examples/experiments/02_optimize.py",
        "examples/extensions/03_numeric_inputs.py",
        "examples/cli/README.md",
    }
    missing = required - relative
    if missing:
        raise RuntimeError(f"Sdist {path.name!r} is missing files: {sorted(missing)!r}.")


def main() -> None:
    directory = Path(sys.argv[1] if len(sys.argv) > 1 else "dist").resolve()
    check_wheel(_single(directory, "*.whl"))
    check_sdist(_single(directory, "*.tar.gz"))
    print(f"Distribution artifacts are complete: {directory}")


if __name__ == "__main__":
    main()
