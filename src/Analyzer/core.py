"""Inference core shared by general and ISA Analyzer compatibility classes."""

from __future__ import annotations

import copy
import os
import pickle
import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from D4CMPP2.src.utils import PATH, module_loader
from D4CMPP2.src.utils.scaler import Scaler, identityScaler

from .results import PredictionResult, PredictionRow, UncertaintyResult


@dataclass(frozen=True)
class ModelArtifacts:
    """Resolved files required to reconstruct a saved model."""

    root: Path
    config: Path
    network: Path
    weights: Path
    scaler: Path | None
    functional_group: Path | None


class _CompatibleScalerUnpickler(pickle.Unpickler):
    """Map only known historical D4CMPP scaler classes to current equivalents."""

    _ALIASES = {
        ("D4CMPP.src.utils.scaler", "Scaler"): Scaler,
        ("D4CMPP.src.utils.scaler", "identityScaler"): identityScaler,
        ("D4CMPP2.src.utils.scaler", "Scaler"): Scaler,
        ("D4CMPP2.src.utils.scaler", "identityScaler"): identityScaler,
    }

    def find_class(self, module, name):
        alias = self._ALIASES.get((module, name))
        if alias is not None:
            return alias
        return super().find_class(module, name)


@contextmanager
def _prevent_saved_model_bytecode():
    """Keep importlib from writing __pycache__ into a saved model folder."""

    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        yield
    finally:
        sys.dont_write_bytecode = previous


def resolve_model_artifacts(model_path, *, model_dir=None) -> ModelArtifacts:
    """Resolve a saved model and report missing artifacts together."""

    config_hint = {"MODEL_DIR": os.fspath(model_dir)} if model_dir is not None else None
    root = Path(PATH.find_model_path(model_path, config_hint)).expanduser().resolve()
    paths = {
        "config.yaml": root / "config.yaml",
        "network.py": root / "network.py",
        "final.pth": root / "final.pth",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Saved model {str(root)!r} is incomplete. Missing required artifacts: {missing}. "
            "Restore the files from the original model folder or provide a complete model path."
        )
    scaler = root / "scaler.pkl"
    functional_group = root / "functional_group.csv"
    return ModelArtifacts(
        root=root,
        config=paths["config.yaml"],
        network=paths["network.py"],
        weights=paths["final.pth"],
        scaler=scaler if scaler.is_file() else None,
        functional_group=functional_group if functional_group.is_file() else None,
    )


def _as_list(value, column):
    if isinstance(value, str):
        return [value]
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return [value.item()]
        if value.ndim != 1:
            raise ValueError(
                f"Input column {column!r} must be one-dimensional, "
                f"got NumPy shape {value.shape}."
            )
        return value.tolist()
    if hasattr(value, "tolist") and not isinstance(value, (list, tuple)):
        converted = value.tolist()
        if isinstance(converted, list) and any(
            isinstance(item, (list, tuple)) for item in converted
        ):
            raise ValueError(
                f"Input column {column!r} must be one-dimensional; "
                f"got nested values from {type(value).__name__}."
            )
        return converted if isinstance(converted, list) else [converted]
    if isinstance(value, (list, tuple)):
        return list(value)
    if np.isscalar(value):
        return [value.item() if hasattr(value, "item") else value]
    raise TypeError(
        f"Input column {column!r} must be a scalar or one-dimensional sequence, "
        f"got {type(value).__name__}."
    )


class InferenceCore:
    """Load one saved model and execute row-preserving inference."""

    def __init__(
        self,
        model_path,
        *,
        device=None,
        batch_size=None,
        model_dir=None,
    ):
        self.artifacts = resolve_model_artifacts(model_path, model_dir=model_dir)
        with self.artifacts.config.open(encoding="utf-8") as file:
            loaded = yaml.load(file, Loader=yaml.FullLoader)
        if not isinstance(loaded, dict):
            raise ValueError(
                f"Saved config {str(self.artifacts.config)!r} must contain a mapping, "
                f"got {type(loaded).__name__}."
            )

        self.config = copy.deepcopy(loaded)
        if "sculptor_index" in self.config and not {
            "sculptor_s",
            "sculptor_c",
            "sculptor_a",
        }.issubset(self.config):
            sculptor_index = self.config["sculptor_index"]
            if not isinstance(sculptor_index, (list, tuple)) or len(sculptor_index) != 3:
                raise ValueError(
                    f"Saved config {str(self.artifacts.config)!r} has invalid sculptor_index "
                    f"{sculptor_index!r}; expected three integers."
                )
            (
                self.config["sculptor_s"],
                self.config["sculptor_c"],
                self.config["sculptor_a"],
            ) = tuple(sculptor_index)
        root = str(self.artifacts.root)
        self.config["MODEL_PATH"] = root
        self.config["LOAD_PATH"] = root
        if self.artifacts.functional_group is not None:
            self.config["FRAG_REF"] = str(self.artifacts.functional_group)
        if device is not None:
            self.config["device"] = str(device)
        if batch_size is not None:
            if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
                raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}.")
            self.config["batch_size"] = batch_size

        self.targets = tuple(self.config.get("target", []))
        if not self.targets:
            raise ValueError(
                f"Saved config {str(self.artifacts.config)!r} has no target columns. "
                "Restore the training config with a non-empty target list."
            )

        is_isa = "ISA" in str(self.config.get("data_manager_class", ""))
        if is_isa and self.artifacts.functional_group is None:
            raise FileNotFoundError(
                f"ISA saved model {root!r} is missing 'functional_group.csv'. "
                "Restore the functional-group file saved during training."
            )

        scaler_name = str(self.config.get("scaler", "identity")).lower()
        if self.artifacts.scaler is None:
            if scaler_name not in {"identity", "none"}:
                raise FileNotFoundError(
                    f"Saved model {root!r} uses scaler {scaler_name!r} but 'scaler.pkl' is missing. "
                    "Restore the fitted scaler; predictions cannot be converted to target units without it."
                )
            self.scaler = None
        else:
            try:
                with self.artifacts.scaler.open("rb") as file:
                    self.scaler = _CompatibleScalerUnpickler(file).load()
            except (
                OSError,
                pickle.PickleError,
                EOFError,
                AttributeError,
                ImportError,
            ) as exc:
                raise ValueError(
                    f"Could not load scaler {str(self.artifacts.scaler)!r}. "
                    "Check that it is the original, compatible scaler.pkl."
                ) from exc

        # DataManager reconstructs the graph contract and may populate feature dimensions
        # in the copied config before the saved network is initialized.
        self.dm = module_loader.load_data_manager(self.config)(self.config)
        self.molecule_columns = tuple(self.dm.molecule_columns)
        self.numeric_input_columns = tuple(self.dm.numeric_input_columns)
        self.input_columns = self.molecule_columns + self.numeric_input_columns
        with _prevent_saved_model_bytecode():
            self.nm = module_loader.load_network_manager(self.config)(
                self.config, unwrapper=self.dm.unwrapper, temp=True
            )
        self.tm = module_loader.load_train_manager(self.config)(self.config)

    def normalize_inputs(self, args, kwargs):
        """Return copied, equally sized input columns without mutating caller data."""

        values = dict(kwargs)
        if args:
            if len(args) > len(self.input_columns):
                raise ValueError(
                    f"Expected at most {len(self.input_columns)} positional inputs in order "
                    f"{list(self.input_columns)}, got {len(args)}."
                )
            for column, value in zip(self.input_columns, args):
                if column in values:
                    raise ValueError(f"Input column {column!r} was provided both positionally and by keyword.")
                values[column] = value

        unknown = sorted(set(values) - set(self.input_columns))
        if unknown:
            raise ValueError(
                f"Unknown input columns {unknown}. Expected molecule columns "
                f"{list(self.molecule_columns)} and numeric columns {list(self.numeric_input_columns)}."
            )
        missing = [column for column in self.input_columns if column not in values]
        if missing:
            raise ValueError(
                f"Missing required input columns {missing}. Expected columns: {list(self.input_columns)}."
            )
        normalized = {column: _as_list(values[column], column) for column in self.input_columns}
        lengths = {column: len(items) for column, items in normalized.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"All Analyzer input columns must have the same length; received lengths {lengths}."
            )
        if next(iter(lengths.values()), 0) == 0:
            raise ValueError("Analyzer input must contain at least one row.")
        for column in self.numeric_input_columns:
            for index, value in enumerate(normalized[column]):
                if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
                    raise TypeError(
                        f"Numeric input {column!r} at row {index} must be a number, "
                        f"got {type(value).__name__}."
                    )
                if not np.isfinite(value):
                    raise ValueError(
                        f"Numeric input {column!r} at row {index} must be finite, got {value!r}."
                    )
        return normalized

    def _prepare(self, normalized):
        self.dm.init_temp_data(**{key: list(value) for key, value in normalized.items()})
        loader = self.dm.get_Dataloaders(temp=True)
        indices = [int(value) for value in np.asarray(self.dm.original_row_indices).tolist()]
        return loader, indices

    def _inverse_transform(self, scores):
        array = scores.detach().cpu().numpy() if isinstance(scores, torch.Tensor) else np.asarray(scores)
        return array if self.scaler is None else self.scaler.inverse_transform(array)

    def _result_from_scores(self, normalized, valid_indices, scores, *, metadata=None):
        row_count = len(next(iter(normalized.values())))
        score_array = np.asarray(scores)
        expected_shape = (len(valid_indices), len(self.targets))
        if score_array.shape != expected_shape:
            raise ValueError(
                f"Analyzer prediction shape {score_array.shape} does not match "
                f"expected {expected_shape} for {len(valid_indices)} valid rows "
                f"and targets {list(self.targets)}."
            )
        score_by_row = {
            row: np.asarray(score_array[index])
            for index, row in enumerate(valid_indices)
        }
        errors = {}
        for item in getattr(self.dm, "graph_errors", []):
            row = int(item.get("row_index"))
            message = (
                f"Invalid molecule in column {item.get('type')!r}: {item.get('smiles')!r}. "
                f"Graph generation failed: {item.get('reason', 'unknown reason')}."
            )
            errors.setdefault(row, []).append(message)
        rows = []
        for row_index in range(row_count):
            inputs = {column: normalized[column][row_index] for column in self.input_columns}
            if row_index in score_by_row:
                rows.append(
                    PredictionRow(
                        row_index=row_index,
                        inputs=inputs,
                        prediction=score_by_row[row_index],
                    )
                )
            else:
                message = " ".join(errors.get(row_index, [])) or (
                    "Input row was filtered during graph preparation. Check molecule syntax "
                    "and all required input columns."
                )
                rows.append(
                    PredictionRow(
                        row_index=row_index,
                        inputs=inputs,
                        prediction=None,
                        status="invalid",
                        error=message,
                    )
                )
        return PredictionResult(tuple(rows), self.targets, metadata or {})

    def predict(self, *args, **kwargs) -> PredictionResult:
        normalized = self.normalize_inputs(args, kwargs)
        loader, valid_indices = self._prepare(normalized)
        if not valid_indices:
            empty = np.empty((0, len(self.targets)), dtype=float)
            return self._result_from_scores(normalized, valid_indices, empty)
        scores, _, _ = self.tm.predict(self.nm, loader, dropout=False)
        return self._result_from_scores(
            normalized,
            valid_indices,
            self._inverse_transform(scores),
            metadata={"model_path": str(self.artifacts.root), "device": self.config.get("device")},
        )

    def predict_uncertainty(self, *args, samples=30, seed=None, **kwargs) -> UncertaintyResult:
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 2:
            raise ValueError(f"samples must be an integer of at least 2, got {samples!r}.")
        dropout_modules = [
            module for module in self.nm.network.modules()
            if module.__class__.__name__.startswith("Dropout")
        ]
        if not dropout_modules:
            raise ValueError(
                f"Model {str(self.artifacts.root)!r} has no Dropout modules; MC-dropout "
                "uncertainty is not available. Use an ensemble of independently trained models instead."
            )
        normalized = self.normalize_inputs(args, kwargs)
        loader, valid_indices = self._prepare(normalized)
        if not valid_indices:
            empty = np.empty((0, len(self.targets)), dtype=float)
            result = self._result_from_scores(normalized, valid_indices, empty)
            return UncertaintyResult(result, result, tuple(), "mc_dropout", seed)

        cpu_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        draws = []
        try:
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            for _ in range(samples):
                scores, _, _ = self.tm.predict(self.nm, loader, dropout=True)
                values = self._inverse_transform(scores)
                draws.append(
                    self._result_from_scores(
                        normalized,
                        valid_indices,
                        values,
                        metadata={"method": "mc_dropout"},
                    )
                )
        finally:
            self.nm.eval()
            torch.random.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
            np.random.set_state(numpy_state)
            random.setstate(python_state)

        stack = np.stack(
            [[row.prediction for row in draw.valid_rows] for draw in draws],
            axis=0,
        )
        mean = self._result_from_scores(
            normalized,
            valid_indices,
            stack.mean(axis=0),
            metadata={"method": "mc_dropout", "samples": samples, "seed": seed},
        )
        std = self._result_from_scores(
            normalized,
            valid_indices,
            stack.std(axis=0, ddof=0),
            metadata={"method": "mc_dropout", "samples": samples, "seed": seed},
        )
        return UncertaintyResult(mean, std, tuple(draws), "mc_dropout", seed)


def predict_ensemble(analyzers, *args, **kwargs) -> UncertaintyResult:
    """Aggregate compatible Analyzer instances into ensemble mean/std."""

    analyzers = list(analyzers)
    if len(analyzers) < 2:
        raise ValueError("Ensemble prediction requires at least two Analyzer instances.")
    cores = [getattr(analyzer, "_core", analyzer) for analyzer in analyzers]
    reference = cores[0]
    for index, core in enumerate(cores[1:], start=1):
        if core.input_columns != reference.input_columns:
            raise ValueError(
                f"Ensemble model {index} input columns {list(core.input_columns)} do not match "
                f"{list(reference.input_columns)}."
            )
        if core.targets != reference.targets:
            raise ValueError(
                f"Ensemble model {index} targets {list(core.targets)} do not match "
                f"{list(reference.targets)}."
            )

    draws = tuple(core.predict(*args, **kwargs) for core in cores)
    reference_rows = draws[0].rows
    for model_index, draw in enumerate(draws[1:], start=1):
        signature = [(row.row_index, row.status, dict(row.inputs)) for row in draw.rows]
        expected = [(row.row_index, row.status, dict(row.inputs)) for row in reference_rows]
        if signature != expected:
            raise ValueError(
                f"Ensemble model {model_index} produced different row validation or ordering. "
                "All ensemble members must accept the same input rows."
            )

    valid_positions = [
        index for index, row in enumerate(reference_rows) if row.status == "ok"
    ]
    stack = np.stack(
        [
            [draw.rows[index].prediction for index in valid_positions]
            for draw in draws
        ],
        axis=0,
    )

    def aggregate(values, statistic):
        valid_lookup = {
            position: np.asarray(values[index])
            for index, position in enumerate(valid_positions)
        }
        rows = []
        for position, reference_row in enumerate(reference_rows):
            rows.append(
                PredictionRow(
                    row_index=reference_row.row_index,
                    inputs=dict(reference_row.inputs),
                    prediction=valid_lookup.get(position),
                    status=reference_row.status,
                    error=reference_row.error,
                )
            )
        return PredictionResult(
            tuple(rows),
            reference.targets,
            {"method": "ensemble", "models": len(draws), "statistic": statistic},
        )

    mean = aggregate(stack.mean(axis=0), "mean")
    std = aggregate(stack.std(axis=0, ddof=0), "std")
    return UncertaintyResult(mean, std, draws, "ensemble", None)
