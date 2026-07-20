"""Row- and index-aligned ISA interpretation results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from rdkit import Chem


@dataclass(frozen=True)
class ISAAnalysisRow:
    """Scores and features aligned to one molecule's atoms or fragments."""

    row_index: int
    inputs: Mapping[str, Any]
    atom_count: int
    fragment_atom_indices: tuple[tuple[int, ...], ...]
    scores: Mapping[str, np.ndarray]
    score_mode: str
    features: Mapping[str, np.ndarray]
    feature_mode: str | None
    prediction: np.ndarray | None = None

    def atom_scores(self, key="positive") -> np.ndarray:
        """Return a score for every atom, expanding fragment scores if needed."""

        if key not in self.scores:
            raise KeyError(f"Score {key!r} is unavailable. Available scores: {list(self.scores)}.")
        values = np.asarray(self.scores[key])
        if self.score_mode == "atom":
            if values.shape[0] != self.atom_count:
                raise ValueError(
                    f"Atom score {key!r} has {values.shape[0]} rows, expected {self.atom_count}."
                )
            return values
        output_shape = (self.atom_count,) + values.shape[1:]
        expanded = np.zeros(output_shape, dtype=values.dtype)
        for fragment_index, atoms in enumerate(self.fragment_atom_indices):
            expanded[list(atoms)] = values[fragment_index]
        return expanded


@dataclass(frozen=True)
class ISAAnalysisResult(Sequence[ISAAnalysisRow]):
    """Valid ISA analysis rows in original input order."""

    rows: tuple[ISAAnalysisRow, ...]
    invalid_rows: tuple[Any, ...] = ()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class ISAInterpreter:
    """Execute score/feature extraction and validate all index boundaries."""

    def __init__(
        self,
        core,
        *,
        score_keys=("positive",),
        feature_keys=(),
        score_mode=None,
    ):
        self.core = core
        self.score_keys = tuple(score_keys)
        self.feature_keys = tuple(feature_keys)
        self.score_mode = score_mode
        sculptor = getattr(getattr(core.dm, "gg", None), "sculptor", None)
        if sculptor is None:
            raise ValueError(
                f"Model {str(core.artifacts.root)!r} does not expose the ISA sculptor "
                "required for fragment-aligned interpretation."
            )
        self.sculptor = sculptor

    def _fragments(self, smiles):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"Could not parse SMILES {smiles!r} for ISA fragmentation.")
        fragments = self.sculptor.fragmentation_with_condition(
            molecule, draw=False, get_index=False
        )
        indices = tuple(tuple(int(index) for index in fragment.atoms) for fragment in fragments)
        flattened = sorted(index for fragment in indices for index in fragment)
        expected = list(range(molecule.GetNumAtoms()))
        if flattened != expected:
            raise ValueError(
                f"ISA fragments for {smiles!r} do not cover each atom exactly once. "
                f"Observed atom indices: {flattened}; expected: {expected}."
            )
        return molecule.GetNumAtoms(), indices

    @staticmethod
    def _numpy_mapping(value, kind):
        if not isinstance(value, dict):
            raise ValueError(
                f"ISA {kind} extraction must return a dictionary, got {type(value).__name__}. "
                "This saved network does not implement the requested interpretation contract."
            )
        return {
            key: item.detach().cpu().numpy() if isinstance(item, torch.Tensor) else np.asarray(item)
            for key, item in value.items()
        }

    @staticmethod
    def _split(mapping, counts, *, skip=()):
        total = sum(counts)
        result = [dict() for _ in counts]
        for key, values in mapping.items():
            if key in skip:
                continue
            array = np.asarray(values)
            if array.shape[0] != total:
                raise ValueError(
                    f"ISA output {key!r} has leading dimension {array.shape[0]}, "
                    f"but aligned rows require {total} ({counts})."
                )
            start = 0
            for row, count in zip(result, counts):
                row[key] = array[start:start + count]
                start += count
        return result

    def analyze(self, *args, include_features=True, **kwargs):
        normalized = self.core.normalize_inputs(args, kwargs)
        loader, valid_indices = self.core._prepare(normalized)
        if not valid_indices:
            prediction = self.core._result_from_scores(
                normalized, valid_indices, np.empty((0, len(self.core.targets)))
            )
            return ISAAnalysisResult((), prediction.invalid_rows)

        draw_column = self.core.molecule_columns[0]
        fragment_rows = []
        for row_index in valid_indices:
            fragment_rows.append(self._fragments(normalized[draw_column][row_index]))
        atom_counts = [item[0] for item in fragment_rows]
        fragment_counts = [len(item[1]) for item in fragment_rows]

        score_output = self._numpy_mapping(
            self.core.tm.get_score(self.core.nm, loader), "score"
        )
        prediction_values = score_output.pop("prediction", None)
        network_name = str(self.core.config.get("network", ""))
        missing_scores = [key for key in self.score_keys if key not in score_output]
        if missing_scores:
            raise ValueError(
                f"Saved network {network_name!r} did not return required ISA scores "
                f"{missing_scores!r}. Returned keys: {list(score_output)!r}."
            )
        score_output = {key: score_output[key] for key in self.score_keys}
        score_mode = self.score_mode or (
            "fragment"
            if network_name in {"GC_model", "ISATPN_model", "ISATPM_model"}
            else "atom"
        )
        score_counts = fragment_counts if score_mode == "fragment" else atom_counts
        score_rows = self._split(score_output, score_counts)

        feature_rows = [dict() for _ in valid_indices]
        feature_mode = None
        if include_features:
            if not self.feature_keys:
                raise ValueError(
                    f"Saved network {network_name!r} does not provide ISATPN hidden features. "
                    "Call analyze_rows(..., include_features=False) for score-only analysis."
                )
            feature_output = self._numpy_mapping(
                self.core.tm.get_feature(self.core.nm, loader), "feature"
            )
            # ISATPN's get_feature branch returns dot/group-node hidden states.
            missing_features = [
                key for key in self.feature_keys if key not in feature_output
            ]
            if missing_features:
                raise ValueError(
                    f"Saved network {network_name!r} did not return required ISATPN "
                    f"features {missing_features!r}. Returned keys: "
                    f"{list(feature_output)!r}."
                )
            feature_values = {
                key: feature_output[key] for key in self.feature_keys
            }
            feature_rows = self._split(feature_values, fragment_counts)
            feature_mode = "fragment"

        if prediction_values is not None:
            prediction_values = self.core._inverse_transform(prediction_values)
            if prediction_values.shape[0] != len(valid_indices):
                raise ValueError(
                    f"ISA prediction output has {prediction_values.shape[0]} rows, "
                    f"expected {len(valid_indices)}."
                )

        rows = []
        for position, row_index in enumerate(valid_indices):
            rows.append(
                ISAAnalysisRow(
                    row_index=row_index,
                    inputs={
                        column: normalized[column][row_index]
                        for column in self.core.input_columns
                    },
                    atom_count=atom_counts[position],
                    fragment_atom_indices=fragment_rows[position][1],
                    scores=score_rows[position],
                    score_mode=score_mode,
                    features=feature_rows[position],
                    feature_mode=feature_mode,
                    prediction=(
                        None if prediction_values is None
                        else np.asarray(prediction_values[position])
                    ),
                )
            )
        prediction_result = self.core._result_from_scores(
            normalized,
            valid_indices,
            (
                prediction_values
                if prediction_values is not None
                else np.zeros((len(valid_indices), len(self.core.targets)))
            ),
        )
        return ISAAnalysisResult(tuple(rows), prediction_result.invalid_rows)
