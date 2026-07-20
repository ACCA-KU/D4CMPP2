"""Versioned full-training checkpoint helpers."""

import os
import random
import uuid
from pathlib import Path

import numpy as np
import torch
from D4CMPP2.exceptions import (
    CheckpointFormatError,
    CheckpointIOError,
    CheckpointLoadError,
    CheckpointNotFoundError,
)


CHECKPOINT_SCHEMA_VERSION = 1


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        try:
            torch.save(payload, staging)
            if not staging.is_file() or staging.stat().st_size == 0:
                raise OSError(
                    f"Checkpoint staging file {str(staging)!r} is missing or empty."
                )
            os.replace(staging, path)
        except OSError as exc:
            raise CheckpointIOError(
                f"Could not atomically save checkpoint {str(path)!r}: {exc}"
            ) from exc
    finally:
        if staging.exists():
            staging.unlink()


def resolve_resume_checkpoint(value):
    requested = Path(os.fspath(value))
    path = requested / "checkpoints" / "latest.ckpt" if requested.is_dir() else requested
    if not path.is_file():
        raise CheckpointNotFoundError(
            f"Full resume checkpoint {str(path)!r} was not found. "
            "Use RESUME_PATH with a model folder containing checkpoints/latest.ckpt, "
            "or use LOAD_PATH for a legacy weight-only final.pth continuation."
        )
    return path


def load_checkpoint(path, device):
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise CheckpointLoadError(
            f"Could not load full resume checkpoint {str(path)!r}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise CheckpointFormatError(
            f"Checkpoint {str(path)!r} must contain a dictionary, got {type(payload).__name__}."
        )
    version = payload.get("checkpoint_schema_version")
    if version != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointFormatError(
            f"Checkpoint {str(path)!r} has checkpoint_schema_version={version!r}; "
            f"this version requires {CHECKPOINT_SCHEMA_VERSION}."
        )
    return payload
