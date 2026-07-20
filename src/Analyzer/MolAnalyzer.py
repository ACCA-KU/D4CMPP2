"""Compatibility Analyzer classes backed by the row-preserving inference core."""

from __future__ import annotations

import hashlib
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from .core import InferenceCore
from .results import PredictionResult, UncertaintyResult


class MolAnalyzer:
    """Load a saved model and predict molecular properties.

    This class preserves the legacy ``predict(smiles, solvent_list=None)`` result
    mapping. New code should use :meth:`predict_rows` when duplicate or invalid
    input rows must be retained.
    """

    def __init__(
        self,
        model_path,
        save_result=False,
        *,
        device=None,
        batch_size=None,
        model_dir=None,
    ):
        self._core = InferenceCore(
            model_path,
            device=device,
            batch_size=batch_size,
            model_dir=model_dir,
        )
        self.model_path = str(self._core.artifacts.root)
        self.data_path = str(self._core.artifacts.root / "data")
        self.save_result = bool(save_result)
        self.config = self._core.config
        self.dm = self._core.dm
        self.nm = self._core.nm
        self.tm = self._core.tm
        self.scaler = self._core.scaler
        self.molecule_columns = list(self._core.molecule_columns)
        self.numeric_input_columns = list(self._core.numeric_input_columns)
        self.data_keys = ["prediction"]
        self.for_pickle = []
        if self.save_result:
            Path(self.data_path).mkdir(parents=True, exist_ok=True)

    def predict_rows(self, *args, **kwargs) -> PredictionResult:
        """Predict while retaining every original row and its validation status."""

        result = self._core.predict(*args, **kwargs)
        self._save_prediction_rows(result)
        return result

    def prepare_temp_data(self, *args, **kwargs):
        """Compatibility bridge for ISA helpers that operate on prepared loaders."""

        normalized = self._core.normalize_inputs(args, kwargs)
        self.dm.init_temp_data(**{key: list(value) for key, value in normalized.items()})
        return self.dm.get_Dataloaders(temp=True), {
            **self.dm.valid_smiles,
            **self.dm.numeric_inputs,
        }

    def predict(self, smiles_list, solvent_list=None, dropout=False):
        """Return the historical SMILES-keyed prediction mapping."""

        if dropout:
            warnings.warn(
                "dropout=True returns one stochastic draw and is deprecated. "
                "Use predict_uncertainty(samples=..., seed=...) for reproducible MC dropout.",
                DeprecationWarning,
                stacklevel=2,
            )
            inputs = self._legacy_inputs(smiles_list, solvent_list)
            uncertainty = self._core.predict_uncertainty(**inputs, samples=2)
            structured = uncertainty.samples[0]
        else:
            structured = self.predict_rows(**self._legacy_inputs(smiles_list, solvent_list))

        result = {}
        for row in structured.valid_rows:
            smiles = row.inputs[self.molecule_columns[0]]
            if solvent_list is None:
                result[smiles] = row.prediction
            else:
                solvent_column = self.molecule_columns[1]
                result[(smiles, row.inputs[solvent_column])] = row.prediction
        return result

    def _legacy_inputs(self, smiles_list, solvent_list):
        inputs = {self.molecule_columns[0]: smiles_list}
        if solvent_list is not None:
            if len(self.molecule_columns) < 2:
                raise ValueError(
                    f"Model {self.model_path!r} does not define a solvent molecule column. "
                    f"Expected columns: {self.molecule_columns}."
                )
            inputs[self.molecule_columns[1]] = solvent_list
        for column in self.numeric_input_columns:
            if column not in inputs:
                raise ValueError(
                    f"Legacy MolAnalyzer.predict cannot infer required numeric input {column!r}. "
                    "Use MolAnalyzer_v2.predict with named inputs."
                )
        return inputs

    def predict_uncertainty(self, *args, samples=30, seed=None, **kwargs) -> UncertaintyResult:
        """Estimate MC-dropout mean/std without changing the default predict output."""

        return self._core.predict_uncertainty(*args, samples=samples, seed=seed, **kwargs)

    def predict_csv(
        self,
        input_path,
        output_path=None,
        *,
        index_col=None,
        uncertainty_samples=None,
        uncertainty_seed=None,
        **read_csv_kwargs,
    ):
        """Predict a CSV while preserving every source row and invalid-row error."""

        source = Path(input_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(
                f"Inference CSV {str(source)!r} does not exist. Provide an existing CSV path."
            )
        try:
            frame = pd.read_csv(source, **read_csv_kwargs)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
            raise ValueError(
                f"Could not read inference CSV {str(source)!r}. "
                "Check its encoding, delimiter, header, and row structure."
            ) from exc
        missing = [column for column in self._core.input_columns if column not in frame.columns]
        if missing:
            raise ValueError(
                f"Inference CSV {str(source)!r} is missing required columns {missing}. "
                f"Available columns: {list(frame.columns)}."
            )
        if index_col is not None and index_col not in frame.columns:
            raise ValueError(
                f"Inference CSV {str(source)!r} does not contain index_col "
                f"{index_col!r}. Available columns: {list(frame.columns)}."
            )
        source_indices = (
            frame[index_col].tolist()
            if index_col is not None
            else frame.index.tolist()
        )
        inputs = {column: frame[column].tolist() for column in self._core.input_columns}
        uncertainty = None
        if uncertainty_samples is None:
            result = self.predict_rows(**inputs)
        else:
            uncertainty = self.predict_uncertainty(
                **inputs,
                samples=uncertainty_samples,
                seed=uncertainty_seed,
            )
            result = uncertainty.mean
        output = result.to_dataframe()
        if uncertainty is not None:
            std_frame = uncertainty.std.to_dataframe()
            for target in self._core.targets:
                output[f"{target}_pred_std"] = std_frame[f"{target}_pred"]
            output["uncertainty_samples"] = uncertainty_samples
        output["row_index"] = source_indices
        destination = (
            Path(output_path).expanduser().resolve()
            if output_path is not None
            else source.with_name(source.stem + "_prediction.csv")
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        output.to_csv(temporary, index=False)
        temporary.replace(destination)
        return str(destination)

    def _save_prediction_rows(self, result):
        if not self.save_result:
            return
        for row in result.valid_rows:
            if not self.numeric_input_columns and len(self.molecule_columns) == 1:
                identity = row.inputs[self.molecule_columns[0]]
            elif not self.numeric_input_columns and len(self.molecule_columns) == 2:
                identity = (
                    f"{row.inputs[self.molecule_columns[0]]}_"
                    f"{row.inputs[self.molecule_columns[1]]}"
                )
            else:
                identity = json.dumps(dict(row.inputs), sort_keys=True, default=str)
            self.save_data(identity, {"prediction": row.prediction})

    def save_data(self, identity, data):
        """Save compatible per-input Analyzer cache entries."""

        if not self.save_result:
            return
        for key, value in data.items():
            if key not in self.data_keys:
                raise ValueError(f"Analyzer cache key must be one of {self.data_keys}, got {key!r}.")
            path = Path(self.data_path) / self.get_file_name(identity, key)
            if key in self.for_pickle:
                with path.open("wb") as file:
                    pickle.dump(value, file)
            else:
                with path.open("wb") as file:
                    np.save(file, np.asarray(value))

    def load_data(self, identity, key):
        if not self.save_result:
            return None
        path = Path(self.data_path) / self.get_file_name(identity, key)
        if not path.is_file():
            return None
        try:
            with path.open("rb") as file:
                return pickle.load(file) if key in self.for_pickle else np.load(file)
        except (OSError, ValueError, EOFError, pickle.PickleError) as exc:
            warnings.warn(
                f"Ignoring unreadable Analyzer cache file {str(path)!r}: {exc}. "
                "The value will be recalculated.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

    def get_file_name(self, identity, key):
        if key not in self.data_keys:
            raise ValueError(f"Analyzer cache key must be one of {self.data_keys}, got {key!r}.")
        digest = hashlib.sha256(str(identity).encode("utf-8")).hexdigest()
        extension = "pickle" if key in self.for_pickle else "np"
        return f"{digest}_{self.data_keys.index(key)}.{extension}"


class MolAnalyzer_v2(MolAnalyzer):
    """Generalized Analyzer for named molecule and numeric input columns."""

    def __init__(self, model_path, save_result=True, **kwargs):
        super().__init__(model_path, save_result=save_result, **kwargs)
        try:
            version = tuple(int(part) for part in str(self.config.get("version", "1.0")).split("."))
        except ValueError as exc:
            raise ValueError(
                f"Saved model version {self.config.get('version')!r} is not a dotted numeric version."
            ) from exc
        if version < (1, 3):
            raise ValueError(
                "MolAnalyzer_v2 requires config version 2.0 or the compatible "
                "historical version 1.3. Use MolAnalyzer for a version 1.0 model."
            )

    def predict(self, *args, dropout=False, **kwargs):
        if dropout:
            warnings.warn(
                "dropout=True returns one stochastic draw and is deprecated. "
                "Use predict_uncertainty(samples=..., seed=...) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            structured = self._core.predict_uncertainty(
                *args, samples=2, **kwargs
            ).samples[0]
        else:
            structured = self.predict_rows(*args, **kwargs)
        return structured.legacy_dict(self._core.input_columns)

    def handle_positional_args(self, args, kwargs):
        """Compatibility parser used by ISA plotting methods.

        Analyzer input keys are normalized separately from plotting options, and
        neither the caller's mapping nor its sequence values are mutated.
        """

        input_kwargs = {}
        other_kwargs = {}
        for key, value in dict(kwargs).items():
            if key in self._core.input_columns:
                input_kwargs[key] = value
            else:
                other_kwargs[key] = value
        normalized = self._core.normalize_inputs(args, input_kwargs)
        return normalized, other_kwargs


class MolAnalyzer_v1p3(MolAnalyzer_v2):
    """Deprecated compatibility name for :class:`MolAnalyzer_v2`."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "MolAnalyzer_v1p3 is deprecated; use MolAnalyzer_v2 or Analyzer(...) instead.",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


def mol_with_atom_index(mol):
    """Return an RDKit molecule labelled with its zero-based atom indices."""

    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    if mol is None:
        raise ValueError("Could not parse molecule for atom-index display.")
    for index in range(mol.GetNumAtoms()):
        mol.GetAtomWithIdx(index).SetProp("molAtomMapNumber", str(index))
    return mol
