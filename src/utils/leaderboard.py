"""Read-only experiment comparison for current and legacy model directories."""

import json
import re
from pathlib import Path

import pandas as pd
import yaml


DEFAULT_METRIC = "val_rmse"
_SUMMARY_PATTERNS = {
    "parameter_count": re.compile(r"#params:\s*(\d+)"),
    "runtime_seconds": re.compile(r"learning_time:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))"),
}


def _model_directories(paths):
    if isinstance(paths, (str, Path)):
        paths = [paths]
    if not paths:
        raise ValueError("At least one model directory or search root is required.")

    discovered = set()
    for value in paths:
        root = Path(value).expanduser()
        if not root.exists():
            raise FileNotFoundError(
                f"Experiment path {str(root)!r} does not exist. "
                "Provide a model directory or a directory containing model folders."
            )
        if root.is_file():
            raise ValueError(
                f"Experiment path {str(root)!r} is a file. Provide its model directory instead."
            )
        if (root / "config.yaml").is_file():
            discovered.add(root.resolve())
            continue
        for config_path in root.rglob("config.yaml"):
            discovered.add(config_path.parent.resolve())
    if not discovered:
        shown = ", ".join(repr(str(Path(value))) for value in paths)
        raise ValueError(
            f"No model directories containing config.yaml were found below {shown}. "
            "Check the search roots or pass model directories directly."
        )
    return sorted(discovered, key=lambda path: str(path).casefold())


def _read_yaml(path):
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(
            f"Could not read model config {str(path)!r}: {exc}. "
            "Check that config.yaml is valid UTF-8 YAML."
        ) from exc
    if not isinstance(value, dict):
        raise ValueError(f"Model config {str(path)!r} must contain a YAML mapping.")
    return value


def _read_manifests(model_path):
    manifests = []
    for path in sorted((model_path / "runs").glob("*/run_manifest.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Could not read run manifest {str(path)!r}: {exc}. "
                "Repair or remove the malformed manifest before comparison."
            ) from exc
        if not isinstance(data, dict):
            raise ValueError(f"Run manifest {str(path)!r} must contain a JSON object.")
        data["_manifest_path"] = str(path)
        manifests.append(data)
    return manifests


def _read_metrics(model_path):
    candidates = [model_path / "result" / "metrics.csv", model_path / "metrics.csv"]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        return None, None
    try:
        frame = pd.read_csv(path, index_col=0)
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        raise ValueError(
            f"Could not read metrics file {str(path)!r}: {exc}. "
            "Check that metrics.csv is a valid CSV file."
        ) from exc
    frame.index = frame.index.map(str)
    return frame, str(path)


def _summary(model_path):
    path = model_path / "model_summary.txt"
    if not path.is_file():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return {}
    values = {}
    for key, pattern in _SUMMARY_PATTERNS.items():
        match = pattern.search(text)
        if match:
            values[key] = float(match.group(1)) if key == "runtime_seconds" else int(match.group(1))
    return values


def _targets(config, metrics):
    configured = config.get("target")
    if isinstance(configured, str):
        configured = [configured]
    targets = [str(value) for value in configured or []]
    if metrics is not None:
        targets.extend(str(value) for value in metrics.index)
    return list(dict.fromkeys(targets)) or [None]


def _base_row(model_path, config, manifest, summary):
    network = manifest.get("network", {}) if manifest else {}
    manifest_config = manifest.get("config", {}) if manifest else {}
    effective = dict(config)
    effective.update(manifest_config if isinstance(manifest_config, dict) else {})
    return {
        "model_path": str(model_path),
        "run_id": manifest.get("run_id") if manifest else "legacy",
        "status": manifest.get("status") if manifest else "legacy",
        "mode": manifest.get("mode") if manifest else "legacy",
        "started_at": manifest.get("started_at") if manifest else None,
        "ended_at": manifest.get("ended_at") if manifest else None,
        "duration_seconds": manifest.get("duration_seconds", summary.get("runtime_seconds")) if manifest else summary.get("runtime_seconds"),
        "network": network.get("id") or effective.get("network_id") or effective.get("network"),
        "data": effective.get("data") or Path(str(effective.get("DATA_PATH", ""))).stem or None,
        "best_epoch": manifest.get("best_epoch") if manifest else None,
        "parameter_count": summary.get("parameter_count"),
        "batch_size": effective.get("batch_size"),
        "learning_rate": effective.get("learning_rate", effective.get("lr")),
        "optimizer": effective.get("optimizer"),
        "max_epoch": effective.get("max_epoch"),
        "manifest_path": manifest.get("_manifest_path") if manifest else None,
    }


def collect_experiments(paths):
    """Return one row per run and target, including legacy model folders."""
    rows = []
    for model_path in _model_directories(paths):
        config = _read_yaml(model_path / "config.yaml")
        manifests = _read_manifests(model_path)
        metrics, metrics_path = _read_metrics(model_path)
        summary = _summary(model_path)
        latest_completed = None
        completed = [item for item in manifests if item.get("status") == "completed"]
        if completed:
            latest_completed = max(
                completed,
                key=lambda item: (item.get("ended_at") or "", item.get("run_id") or ""),
            )
        entries = manifests or [None]
        for manifest in entries:
            attach_metrics = manifest is None or manifest is latest_completed
            for target in _targets(config, metrics if attach_metrics else None):
                row = _base_row(model_path, config, manifest, summary)
                row["target"] = target
                row["metric_source"] = metrics_path if attach_metrics else None
                if attach_metrics and metrics is not None and target in metrics.index:
                    for name, value in metrics.loc[target].items():
                        row[str(name)] = value
                rows.append(row)
    return pd.DataFrame(rows)


def compare_experiments(paths, output_path="leaderboard.csv", metric=DEFAULT_METRIC, target=None, ascending=None):
    """Collect, rank within each target, save CSV, and return a DataFrame."""
    frame = collect_experiments(paths)
    if target is not None:
        frame = frame[frame["target"] == str(target)].copy()
        if frame.empty:
            available = sorted(value for value in collect_experiments(paths)["target"].dropna().unique())
            raise ValueError(
                f"Target {target!r} was not found. Available targets: {available}."
            )
    if metric not in frame.columns:
        available = sorted(
            column for column in frame.columns
            if column.startswith(("train_", "val_", "test_"))
        )
        raise ValueError(
            f"Metric {metric!r} was not found. Available metric columns: {available}."
        )
    if ascending is None:
        ascending = not metric.endswith(("_r2", "_accuracy"))
    frame["selected_metric"] = metric
    frame["selected_metric_value"] = pd.to_numeric(frame[metric], errors="coerce")
    frame["rank"] = frame.groupby("target", dropna=False)["selected_metric_value"].rank(
        method="min",
        ascending=ascending,
        na_option="bottom",
    )
    frame = frame.sort_values(
        ["target", "rank", "model_path", "run_id"],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return frame
