"""Additive, best-effort run manifest recording."""

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path


MANIFEST_SCHEMA_VERSION = 1


def _json_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            name = str(key)
            if any(marker in name.lower() for marker in ("password", "token", "secret", "credential")):
                result[name] = "<redacted>"
            else:
                result[name] = _json_safe(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _version(module_name):
    try:
        module = __import__(module_name)
        return str(getattr(module, "__version__", "unknown"))
    except (ImportError, OSError):
        return None


def _file_hash(path):
    if not path or not Path(path).is_file():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(root):
    try:
        commit = subprocess.run(
            ["git", "-c", f"safe.directory={root}", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-c", f"safe.directory={root}", "-C", str(root), "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


class RunManifest:
    def __init__(self, config, mode):
        self.started = time.time()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        self.run_id = f"{stamp}-{uuid.uuid4().hex[:12]}"
        self.path = Path(config["MODEL_PATH"]) / "runs" / self.run_id / "run_manifest.json"
        package_root = Path(__file__).resolve().parents[2]
        self.data = {
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "run_id": self.run_id,
            "status": "running",
            "mode": mode,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "config": _json_safe(config),
            "network": {
                "id": config.get("network_id", config.get("network")),
                "module": config.get("network"),
                "data_manager": config.get("data_manager_class"),
                "network_manager": config.get("network_manager_class"),
                "train_manager": config.get("train_manager_class"),
            },
            "data": {
                "path": config.get("DATA_PATH"),
                "sha256": _file_hash(config.get("DATA_PATH")),
                "targets": _json_safe(config.get("target")),
            },
            "graph": {
                "backend": "pyg",
                "schema_version": 2,
                "cache_directory": config.get("GRAPH_DIR"),
                "cache_policy": config.get("graph_cache_policy", "v2"),
            },
            "seeds": {
                "split_random_seed": config.get("split_random_seed", 42),
                "random_seed": config.get("random_seed"),
                "scheduler_policy": config.get("scheduler_policy", "legacy_dual"),
                "effective_reproducibility": _json_safe(
                    config.get("effective_reproducibility", {})
                ),
            },
            "split": {
                "strategy": config.get("split_strategy", "auto"),
                "scaffold_column": config.get("scaffold_column"),
                "scaffold_include_chirality": config.get(
                    "scaffold_include_chirality", False
                ),
            },
            "environment": {
                "python": sys.version.split()[0],
                "torch": _version("torch"),
                "torch_geometric": _version("torch_geometric"),
                "rdkit": _version("rdkit"),
                "numpy": _version("numpy"),
                "pandas": _version("pandas"),
                "os": platform.platform(),
                "device": config.get("device"),
                "git": _git_state(package_root),
            },
        }
        self.write()

    def update(self, **values):
        self.data.update(_json_safe(values))
        self.write()

    def finish(self, status, error=None, **values):
        self.data.update(_json_safe(values))
        self.data["status"] = status
        self.data["ended_at"] = datetime.now(timezone.utc).isoformat()
        self.data["duration_seconds"] = time.time() - self.started
        if error is not None:
            self.data["error"] = {
                "type": type(error).__name__,
                "message": str(error)[:2000],
            }
        self.write()

    def write(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            staging = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
            try:
                staging.write_text(
                    json.dumps(self.data, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                os.replace(staging, self.path)
            finally:
                if staging.exists():
                    staging.unlink()
        except OSError as exc:
            warnings.warn(
                f"Run manifest {str(self.path)!r} could not be written: {exc}. "
                "Training/checkpoint results are unaffected.",
                RuntimeWarning,
                stacklevel=2,
            )
