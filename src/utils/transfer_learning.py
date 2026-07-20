"""Transfer-learning state selection and audit reporting."""

import hashlib
import json
import os
import uuid
from pathlib import Path

import torch


TRANSFER_REPORT_SCHEMA_VERSION = 1


def _tensor_description(tensor):
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
    }


def select_compatible_state(source_state, target_state):
    """Return name/shape-compatible source state and a complete audit report."""
    if not isinstance(source_state, dict):
        raise TypeError(
            "Transfer checkpoint must contain a model state-dict mapping parameter "
            f"names to tensors, got {type(source_state).__name__}."
        )

    non_tensors = [
        name for name, value in source_state.items() if not isinstance(value, torch.Tensor)
    ]
    if non_tensors:
        raise TypeError(
            "Transfer checkpoint contains non-tensor state entries "
            f"{non_tensors[:10]!r}. Check that final.pth is a plain model state dict."
        )

    selected = {}
    loaded = []
    shape_mismatch = []
    source_only = []
    for name, source_value in source_state.items():
        target_value = target_state.get(name)
        if target_value is None:
            source_only.append({"name": name, **_tensor_description(source_value)})
        elif source_value.shape != target_value.shape:
            shape_mismatch.append(
                {
                    "name": name,
                    "source_shape": list(source_value.shape),
                    "target_shape": list(target_value.shape),
                    "source_dtype": str(source_value.dtype),
                    "target_dtype": str(target_value.dtype),
                }
            )
        else:
            selected[name] = source_value
            loaded.append(
                {
                    "name": name,
                    "shape": list(source_value.shape),
                    "source_dtype": str(source_value.dtype),
                    "target_dtype": str(target_value.dtype),
                }
            )

    target_only = [
        {"name": name, **_tensor_description(value)}
        for name, value in target_state.items()
        if name not in source_state
    ]
    report = {
        "transfer_report_schema_version": TRANSFER_REPORT_SCHEMA_VERSION,
        "counts": {
            "loaded": len(loaded),
            "shape_mismatch": len(shape_mismatch),
            "source_only": len(source_only),
            "target_only": len(target_only),
        },
        "loaded": loaded,
        "shape_mismatch": shape_mismatch,
        "source_only": source_only,
        "target_only": target_only,
    }
    return selected, report


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_transfer_report(report, model_path):
    """Atomically write the additive transfer audit artifact."""
    path = Path(model_path) / "transfer_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        staging.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(staging, path)
    finally:
        if staging.exists():
            staging.unlink()
    return str(path)
