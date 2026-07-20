# Test suite

The default suite is dependency-light and does not import D4CMPP2, PyTorch,
PyTorch Geometric, RDKit, pandas, or scikit-learn. From the workspace root (the directory
that contains both `.venv` and `D4CMPP2`), run:

```powershell
.venv\Scripts\python.exe -m unittest discover -s D4CMPP2\tests -v
```

Alternatively, change to the repository root first:

```powershell
Set-Location D4CMPP2
..\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests marked with the `heavy_test` decorator are skipped by default. Enable
them only in an environment containing the project ML dependencies:

```powershell
$env:D4CMPP2_RUN_HEAVY = "1"
.venv\Scripts\python.exe -m unittest discover -s D4CMPP2\tests -v
```

The GCN end-to-end test exercises the implementation's PyG path.
`tests/baselines/gcn_cpu.json` remains `pending` until the workspace `.venv` is
installed and the test has produced reproducible CPU predictions. A future
Graph-backend changes are compatibility changes; update the test only
alongside its approved cache/model migration policy.

All tests that may create `_Models`, `_Graphs`, `_XYZ`, or other runtime files
must use `isolated_workdir()` from `tests/fixtures.py`. The context manager
changes into a fresh temporary directory and restores the original working
directory even when a test fails.

The committed `data/tiny_regression.csv` fixture is intentionally small and
contains no private or downloaded data. It has 12 rows so it can later exercise
the package's minimum-size automatic split boundary.
