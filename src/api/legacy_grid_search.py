"""Compatibility-preserving exhaustive grid search."""

import copy
import csv
import itertools
import json
import os
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from D4CMPP2.src.api.training import check_args, run, set_config
from D4CMPP2.src.utils import supportfile_saver
from D4CMPP2.src.utils.output import get_output


GRID_SUMMARY_SCHEMA_VERSION = 1


def grid_generator(param_grid):
    if not isinstance(param_grid, dict) or not param_grid:
        raise ValueError(
            "hyperparameters must be a non-empty mapping of parameter names to "
            "non-empty value sequences."
        )
    invalid = {
        key: values
        for key, values in param_grid.items()
        if (
            not isinstance(key, str)
            or not key.strip()
            or isinstance(values, (str, bytes))
            or not isinstance(values, (list, tuple))
            or not values
        )
    }
    if invalid:
        raise ValueError(
            "Each hyperparameter key must be a non-empty string and each value "
            f"must be a non-empty list or tuple; invalid entries: {invalid!r}."
        )
    keys, values = zip(*param_grid.items())
    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))


config0 = {
    "data": None,
    "target": [],
    "network": None,
    "scaler": "standard",
    "optimizer": "Adam",
    "max_epoch": 2000,
    "batch_size": 256,
    "learning_rate": 0.001,
    "weight_decay": 0.0005,
    "lr_patience": 40,
    "early_stopping_patience": 100,
    "device": "cuda:0",
    "pin_memory": False,
}


def _grid_suffix(parameters):
    return "".join(f"_{key},{value}" for key, value in parameters.items())


def _unique_trial_path(base_path, parameters, trial_number, used_paths):
    candidate = f"{base_path}{_grid_suffix(parameters)}"
    normalized = os.path.normcase(os.path.abspath(candidate))
    if normalized not in used_paths and not os.path.exists(candidate):
        used_paths.add(normalized)
        return candidate

    candidate = f"{candidate}__trial_{trial_number:04d}"
    normalized = os.path.normcase(os.path.abspath(candidate))
    suffix = 1
    while normalized in used_paths or os.path.exists(candidate):
        candidate = (
            f"{base_path}{_grid_suffix(parameters)}"
            f"__trial_{trial_number:04d}_{suffix}"
        )
        normalized = os.path.normcase(os.path.abspath(candidate))
        suffix += 1
    used_paths.add(normalized)
    return candidate


def _atomic_json(path, value):
    staging = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        staging.write_text(
            json.dumps(value, indent=2, sort_keys=True, default=repr),
            encoding="utf-8",
        )
        os.replace(staging, path)
    finally:
        if staging.exists():
            staging.unlink()


def _write_summary(summary, base_path):
    base = Path(base_path)
    json_path = base.parent / f"{base.name}_grid_search.json"
    csv_path = base.parent / f"{base.name}_grid_search.csv"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    summary["summary_json"] = str(json_path)
    summary["summary_csv"] = str(csv_path)
    _atomic_json(json_path, summary)

    fieldnames = [
        "trial",
        "status",
        "model_path",
        "parameters",
        "started_at",
        "ended_at",
        "duration_seconds",
        "error_type",
        "error_message",
    ]
    staging = csv_path.with_name(f".{csv_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(staging, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for trial in summary["trials"]:
                row = {key: trial.get(key) for key in fieldnames}
                row["parameters"] = json.dumps(
                    trial["parameters"], sort_keys=True, default=repr
                )
                writer.writerow(row)
        os.replace(staging, csv_path)
    finally:
        if staging.exists():
            staging.unlink()
    return str(json_path), str(csv_path)


def grid_search(hyperparameters, **kwargs):
    """Run every parameter combination and write a JSON/CSV status summary.

    The historical return value remains ``None``. Hyperparameter values override
    both base arguments and registry defaults for each isolated trial.
    """
    combinations = list(grid_generator(copy.deepcopy(hyperparameters)))
    base_kwargs = copy.deepcopy(config0)
    base_kwargs.update(copy.deepcopy(kwargs))
    base_config = set_config(**check_args(**base_kwargs))
    output = get_output(base_config)
    base_model_path = base_config["MODEL_PATH"]
    started = datetime.now(timezone.utc)
    summary = {
        "grid_summary_schema_version": GRID_SUMMARY_SCHEMA_VERSION,
        "status": "running",
        "started_at": started.isoformat(),
        "base_model_path": base_model_path,
        "hyperparameters": copy.deepcopy(hyperparameters),
        "trial_count": len(combinations),
        "completed_count": 0,
        "failed_count": 0,
        "interrupted_count": 0,
        "trials": [],
    }
    used_paths = set()
    interrupted_at = None

    for trial_number, parameters in enumerate(combinations, start=1):
        trial_started = datetime.now(timezone.utc)
        trial_config = copy.deepcopy(base_config)
        trial_config.update(copy.deepcopy(parameters))
        trial_path = _unique_trial_path(
            base_model_path, parameters, trial_number, used_paths
        )
        trial_config["MODEL_PATH"] = trial_path
        trial = {
            "trial": trial_number,
            "status": "running",
            "parameters": copy.deepcopy(parameters),
            "model_path": trial_path,
            "started_at": trial_started.isoformat(),
        }
        summary["trials"].append(trial)
        _write_summary(summary, base_model_path)
        try:
            Path(trial_path).mkdir(parents=True, exist_ok=False)
            supportfile_saver.save_additional_files(trial_config)
            result_path = run(trial_config)
            trial["status"] = "completed"
            trial["result_path"] = result_path
            summary["completed_count"] += 1
        except KeyboardInterrupt:
            trial["status"] = "interrupted"
            summary["interrupted_count"] += 1
            interrupted_at = trial_number
            output.error(
                f"[Grid Search] Interrupted during trial {trial_number}."
            )
            break
        except Exception as exc:
            trial["status"] = "failed"
            trial["error_type"] = type(exc).__name__
            trial["error_message"] = str(exc)[:2000]
            trial["traceback"] = traceback.format_exc()[-8000:]
            summary["failed_count"] += 1
            output.error(
                f"[Grid Search] Trial {trial_number} failed:\n"
                f"{trial['traceback']}"
            )
        finally:
            ended = datetime.now(timezone.utc)
            trial["ended_at"] = ended.isoformat()
            trial["duration_seconds"] = (
                ended - trial_started
            ).total_seconds()
            _write_summary(summary, base_model_path)

    if interrupted_at is not None:
        for trial_number, parameters in enumerate(
            combinations[interrupted_at:], start=interrupted_at + 1
        ):
            summary["trials"].append(
                {
                    "trial": trial_number,
                    "status": "not_started",
                    "parameters": copy.deepcopy(parameters),
                    "model_path": None,
                    "error_type": None,
                    "error_message": "Grid search was interrupted before this trial started.",
                }
            )

    ended = datetime.now(timezone.utc)
    if summary["interrupted_count"]:
        summary["status"] = "interrupted"
    elif summary["failed_count"]:
        summary["status"] = (
            "failed" if summary["completed_count"] == 0 else "completed_with_failures"
        )
    else:
        summary["status"] = "completed"
    summary["ended_at"] = ended.isoformat()
    summary["duration_seconds"] = (ended - started).total_seconds()
    json_path, csv_path = _write_summary(summary, base_model_path)
    output.always(
        "[Grid Search] Run complete: "
        f"completed={summary['completed_count']}, "
        f"failed={summary['failed_count']}, "
        f"interrupted={summary['interrupted_count']}. "
        f"JSON: {json_path!r}; CSV: {csv_path!r}."
    )
    return None


if __name__ == "__main__":
    raise SystemExit(
        "Call grid_search(hyperparameters, **training_kwargs) from Python or D4CMPP2."
    )
