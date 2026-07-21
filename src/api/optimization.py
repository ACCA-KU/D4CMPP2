"""Model-owned grid and Gaussian-process hyperparameter optimization."""

import csv
import itertools
import json
import math
import os
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from D4CMPP2.src.api.training import train
from D4CMPP2.networks.base import Hyperparameter
from D4CMPP2.networks.registry import get_model


OPTIMIZATION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class OptimizationResult:
    best_params: dict
    best_score: float
    best_model_path: str
    summary_path: str
    trials: tuple[dict, ...]


@dataclass(frozen=True)
class _Domain:
    name: str
    kind: str
    values: tuple = ()
    low: float | int | None = None
    high: float | int | None = None
    step: float | int | None = None
    scale: str = "linear"

    def grid_values(self):
        if self.values:
            return self.values
        if self.step is None:
            raise ValueError(
                f"Grid domain {self.name!r} needs a value list or a step."
            )
        count = int(math.floor((self.high - self.low) / self.step)) + 1
        values = tuple(self.low + index * self.step for index in range(count))
        if self.kind == "int":
            values = tuple(int(value) for value in values)
        return values

    def sample(self, rng):
        if self.values:
            return self.values[int(rng.integers(0, len(self.values)))]
        if self.scale == "log":
            value = math.exp(rng.uniform(math.log(self.low), math.log(self.high)))
        else:
            value = rng.uniform(self.low, self.high)
        if self.kind == "int":
            step = int(self.step or 1)
            value = int(round((value - self.low) / step) * step + self.low)
            return min(int(self.high), max(int(self.low), value))
        if self.step:
            value = round((value - self.low) / self.step) * self.step + self.low
        return min(float(self.high), max(float(self.low), float(value)))

    def encode(self, value):
        if self.values:
            index = self.values.index(value)
            return 0.0 if len(self.values) == 1 else index / (len(self.values) - 1)
        if self.scale == "log":
            low, high, current = math.log(self.low), math.log(self.high), math.log(value)
        else:
            low, high, current = self.low, self.high, value
        return 0.0 if high == low else (current - low) / (high - low)


def _from_metadata(name, field, strategy):
    if strategy == "grid":
        if not field.grid:
            raise ValueError(
                f"{name!r} has no predefined grid for this model. "
                "Pass HP as a dict with explicit values."
            )
        return _Domain(name, field.kind, values=tuple(field.grid))
    if field.kind == "categorical":
        values = field.values or field.grid
        return _Domain(name, field.kind, values=tuple(values))
    if field.search_low is None or field.search_high is None:
        raise ValueError(
            f"{name!r} has no predefined Bayesian range. "
            "Pass HP as a dict with low/high."
        )
    return _Domain(
        name,
        field.kind,
        low=field.search_low,
        high=field.search_high,
        step=field.step,
        scale=field.scale,
    )


def _explicit_domain(name, specification, field):
    if isinstance(specification, list):
        if not specification:
            raise ValueError(f"HP[{name!r}] grid cannot be empty.")
        values = tuple(field.validate(name, value) for value in specification)
        return _Domain(name, field.kind, values=values)
    if isinstance(specification, tuple):
        if len(specification) != 2:
            raise ValueError(
                f"HP[{name!r}] range tuple must be (low, high), got {specification!r}."
            )
        specification = {"low": specification[0], "high": specification[1]}
    if not isinstance(specification, dict):
        raise TypeError(
            f"HP[{name!r}] must be a list grid, (low, high) tuple, or range dict."
        )
    if "values" in specification or "grid" in specification:
        values = specification.get("values", specification.get("grid"))
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"HP[{name!r}] values/grid must be non-empty.")
        return _Domain(
            name,
            field.kind,
            values=tuple(field.validate(name, value) for value in values),
        )
    if "low" not in specification or "high" not in specification:
        raise ValueError(
            f"HP[{name!r}] range requires low and high, or values/grid."
        )
    if field.kind == "categorical":
        raise ValueError(
            f"HP[{name!r}] is categorical; pass a non-empty values/grid list."
        )
    low = field.validate(name, specification["low"])
    high = field.validate(name, specification["high"])
    if low >= high:
        raise ValueError(f"HP[{name!r}] requires low < high, got {low!r}, {high!r}.")
    scale = specification.get("scale", field.scale)
    if scale not in {"linear", "log"}:
        raise ValueError(f"HP[{name!r}] scale must be 'linear' or 'log'.")
    if scale == "log" and low <= 0:
        raise ValueError(f"HP[{name!r}] log range requires low > 0.")
    step = specification.get("step", field.step)
    if step is not None:
        if isinstance(step, bool) or not isinstance(step, (int, float)) or step <= 0:
            raise ValueError(f"HP[{name!r}] step must be a positive number.")
        if field.kind == "int" and not isinstance(step, int):
            raise TypeError(f"HP[{name!r}] integer range requires an integer step.")
    return _Domain(
        name,
        field.kind,
        low=low,
        high=high,
        step=step,
        scale=scale,
    )


def normalize_hp(network, HP, strategy):
    if strategy not in {"grid", "bayesian"}:
        raise ValueError(
            f"optimize_strategy must be 'grid' or 'bayesian', got {strategy!r}."
        )
    definition = get_model(network)
    model = definition.network
    if HP is None:
        selected = model.optimization_space()
        return tuple(
            _from_metadata(name, field, strategy)
            for name, field in selected.items()
        )
    if isinstance(HP, list):
        if not HP or not all(isinstance(name, str) and name for name in HP):
            raise ValueError("HP list must contain one or more non-empty keys.")
        selected = model.optimization_space(HP)
        return tuple(
            _from_metadata(name, field, strategy)
            for name, field in selected.items()
        )
    if not isinstance(HP, dict) or not HP:
        raise TypeError("HP must be None, a non-empty key list, or a non-empty dict.")
    unknown = [name for name in HP if name not in model.hyperparameters]
    if unknown:
        raise ValueError(
            f"{model.model_name} does not define hyperparameters {unknown!r}. "
            f"Available keys: {sorted(model.hyperparameters)!r}."
        )
    return tuple(
        _explicit_domain(name, specification, model.hyperparameters[name])
        for name, specification in HP.items()
    )


def _signature(parameters):
    return json.dumps(parameters, sort_keys=True, separators=(",", ":"), default=repr)


def _domain_records(domains):
    return [
        {
            "name": domain.name,
            "kind": domain.kind,
            "values": list(domain.values),
            "low": domain.low,
            "high": domain.high,
            "step": domain.step,
            "scale": domain.scale,
        }
        for domain in domains
    ]


def _read_objective(model_path):
    candidates = (
        Path(model_path) / "result" / "learning_curve.csv",
        Path(model_path) / "learning_curve.csv",
    )
    for path in candidates:
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8", newline="") as stream:
            values = [
                float(row["val_loss"])
                for row in csv.DictReader(stream)
                if row.get("val_loss") not in (None, "")
            ]
        finite = [value for value in values if math.isfinite(value)]
        if finite:
            return min(finite)
    raise FileNotFoundError(
        f"No finite val_loss was found below model path {str(model_path)!r}."
    )


def _atomic_json(path, value):
    staging = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        staging.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(staging, path)
    finally:
        if staging.exists():
            staging.unlink()


def _write_summary(path, summary):
    _atomic_json(path, summary)
    csv_path = path.with_suffix(".csv")
    staging = csv_path.with_name(f".{csv_path.name}.{uuid.uuid4().hex}.tmp")
    fields = ("trial", "status", "objective", "model_path", "parameters", "error")
    try:
        with open(staging, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for trial in summary["trials"]:
                row = {name: trial.get(name) for name in fields}
                row["parameters"] = json.dumps(trial["parameters"], sort_keys=True)
                writer.writerow(row)
        os.replace(staging, csv_path)
    finally:
        if staging.exists():
            staging.unlink()


def _random_parameters(domains, rng):
    return {domain.name: domain.sample(rng) for domain in domains}


def _bayesian_parameters(domains, completed, used, rng):
    initial_count = max(4, 2 * len(domains))
    if len(completed) < initial_count:
        for _ in range(1000):
            parameters = _random_parameters(domains, rng)
            if _signature(parameters) not in used:
                return parameters
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel

    x = np.asarray(
        [[domain.encode(trial["parameters"][domain.name]) for domain in domains]
         for trial in completed],
        dtype=float,
    )
    y = np.asarray([trial["objective"] for trial in completed], dtype=float)
    surrogate = GaussianProcessRegressor(
        kernel=Matern(nu=2.5) + WhiteKernel(noise_level=1e-6),
        normalize_y=True,
        random_state=0,
        n_restarts_optimizer=1,
    )
    surrogate.fit(x, y)
    candidates = []
    for _ in range(2048):
        parameters = _random_parameters(domains, rng)
        signature = _signature(parameters)
        if signature not in used:
            candidates.append(parameters)
        if len(candidates) >= 512:
            break
    if not candidates:
        raise StopIteration("The finite hyperparameter space is exhausted.")
    encoded = np.asarray(
        [[domain.encode(candidate[domain.name]) for domain in domains]
         for candidate in candidates],
        dtype=float,
    )
    mean, std = surrogate.predict(encoded, return_std=True)
    return candidates[int(np.argmin(mean - 1.96 * std))]


def optimize(
    *,
    data,
    target,
    network,
    HP=None,
    optimize_strategy="bayesian",
    n_trials=None,
    random_seed=42,
    optimization_path=None,
    resume=True,
    **train_kwargs,
):
    """Tune one registered network and return its best completed trial."""
    if not isinstance(optimize_strategy, str):
        raise TypeError("optimize_strategy must be 'grid' or 'bayesian'.")
    strategy = optimize_strategy.lower()
    domains = normalize_hp(network, HP, strategy)
    domain_records = _domain_records(domains)
    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise TypeError("random_seed must be an integer.")
    if strategy == "bayesian":
        n_trials = 20 if n_trials is None else n_trials
        if isinstance(n_trials, bool) or not isinstance(n_trials, int) or n_trials < 1:
            raise ValueError("n_trials must be a positive integer for Bayesian search.")
    elif n_trials is not None:
        raise ValueError("n_trials is only used with optimize_strategy='bayesian'.")

    root = Path(
        optimization_path
        or Path(train_kwargs.get("MODEL_DIR", "_Models"))
        / f"optimize_{network}"
    ).resolve()
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "optimization.json"
    if resume and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("strategy") != strategy or summary.get("network") != network:
            raise ValueError(
                f"Existing optimization summary {str(summary_path)!r} belongs to "
                f"network={summary.get('network')!r}, strategy={summary.get('strategy')!r}."
            )
        if summary.get("domains") != domain_records:
            raise ValueError(
                f"Existing optimization summary {str(summary_path)!r} uses a different "
                "HP search space. Choose another optimization_path or set resume=False."
            )
    else:
        summary = {
            "schema_version": OPTIMIZATION_SCHEMA_VERSION,
            "status": "running",
            "network": network,
            "strategy": strategy,
            "domains": domain_records,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "trials": [],
        }

    completed = [trial for trial in summary["trials"] if trial["status"] == "completed"]
    used = {_signature(trial["parameters"]) for trial in summary["trials"]}
    rng = np.random.default_rng(random_seed + len(summary["trials"]))
    if strategy == "grid":
        names = [domain.name for domain in domains]
        candidates = (
            dict(zip(names, values))
            for values in itertools.product(*(domain.grid_values() for domain in domains))
        )
    else:
        candidates = None

    while True:
        if strategy == "grid":
            parameters = next(
                (candidate for candidate in candidates if _signature(candidate) not in used),
                None,
            )
            if parameters is None:
                break
        else:
            if len(summary["trials"]) >= n_trials:
                break
            try:
                parameters = _bayesian_parameters(domains, completed, used, rng)
            except StopIteration:
                break

        number = len(summary["trials"]) + 1
        model_path = root / "trials" / f"trial_{number:04d}"
        trial = {
            "trial": number,
            "status": "running",
            "parameters": parameters,
            "model_path": str(model_path),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        summary["trials"].append(trial)
        used.add(_signature(parameters))
        _write_summary(summary_path, summary)
        try:
            trial_kwargs = dict(train_kwargs)
            trial_kwargs.update(parameters)
            trial_kwargs.update(
                data=data,
                target=target,
                network=network,
                MODEL_PATH=str(model_path),
            )
            result_path = train(**trial_kwargs)
            trial["model_path"] = result_path
            trial["objective"] = _read_objective(result_path)
            trial["status"] = "completed"
            completed.append(trial)
        except KeyboardInterrupt:
            trial["status"] = "interrupted"
            trial["error"] = "KeyboardInterrupt"
            summary["status"] = "interrupted"
            trial["ended_at"] = datetime.now(timezone.utc).isoformat()
            _write_summary(summary_path, summary)
            raise
        except Exception as exc:
            trial["status"] = "failed"
            trial["error"] = f"{type(exc).__name__}: {exc}"
            trial["traceback"] = traceback.format_exc()[-8000:]
        trial["ended_at"] = datetime.now(timezone.utc).isoformat()
        _write_summary(summary_path, summary)

    if not completed:
        summary["status"] = "failed"
        _write_summary(summary_path, summary)
        failures = "; ".join(
            f"trial {trial['trial']}: {trial.get('error', trial['status'])}"
            for trial in summary["trials"][-3:]
        )
        raise RuntimeError(
            "Optimization completed no successful trials. "
            f"Recent failures: {failures}. See {str(summary_path)!r}."
        )
    best = min(completed, key=lambda trial: trial["objective"])
    summary["status"] = (
        "completed_with_failures"
        if any(trial["status"] == "failed" for trial in summary["trials"])
        else "completed"
    )
    summary["best_trial"] = best["trial"]
    summary["best_objective"] = best["objective"]
    summary["best_parameters"] = best["parameters"]
    summary["best_model_path"] = best["model_path"]
    summary["ended_at"] = datetime.now(timezone.utc).isoformat()
    _write_summary(summary_path, summary)
    return OptimizationResult(
        best_params=dict(best["parameters"]),
        best_score=float(best["objective"]),
        best_model_path=best["model_path"],
        summary_path=str(summary_path),
        trials=tuple(summary["trials"]),
    )
