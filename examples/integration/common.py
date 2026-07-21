"""Shared helpers for isolated public-API integration examples."""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT.parent))


@contextlib.contextmanager
def workflow_workspace(name: str, output_root: str | None):
    """Run in an explicit output folder or a self-cleaning temporary folder."""

    original = Path.cwd()
    if output_root is None:
        with tempfile.TemporaryDirectory(prefix=f"d4cmpp2-{name}-") as temporary:
            root = Path(temporary).resolve()
            os.chdir(root)
            try:
                yield root
            finally:
                os.chdir(original)
        return

    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.chdir(root)
    try:
        yield root
    finally:
        os.chdir(original)


def write_tiny_dataset(root: Path) -> Path:
    """Write one small predefined-split dataset shared by all workflows."""

    path = root / "tiny_integration.csv"
    path.write_text(
        "\n".join(
            (
                "compound,solvent,target_a,target_b,set",
                "CC,O,1.0,12.0,train",
                "CCC,CO,2.0,11.0,train",
                "CCO,O,3.0,10.0,train",
                "CCN,CO,4.0,9.0,train",
                "c1ccccc1,O,5.0,8.0,train",
                "CCCl,CCO,6.0,7.0,train",
                "CCBr,O,7.0,6.0,train",
                "CC(=O)O,CO,8.0,5.0,train",
                "C1CCCCC1,O,9.0,4.0,val",
                "COC,CCO,10.0,3.0,val",
                "CN,O,11.0,2.0,test",
                "CC(C)O,CO,12.0,1.0,test",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def base_train_kwargs(root: Path, data_path: Path) -> dict:
    """Return small deterministic settings while using only public train options."""

    model_dir = root / "models"
    graph_dir = root / "graphs"
    model_dir.mkdir(exist_ok=True)
    graph_dir.mkdir(exist_ok=True)
    return {
        "data": str(data_path),
        "device": "cpu",
        "max_epoch": 1,
        "batch_size": 4,
        "hidden_dim": 8,
        "conv_layers": 1,
        "linear_layers": 1,
        "dropout": 0.0,
        "random_seed": 42,
        "num_workers": 0,
        "MODEL_DIR": str(model_dir),
        "GRAPH_DIR": str(graph_dir),
        "NET_DIR": str(PACKAGE_ROOT / "networks"),
        "NET_REFER": str(PACKAGE_ROOT / "network_refer.yaml"),
        "save_prediction": False,
        "verbose": False,
    }
