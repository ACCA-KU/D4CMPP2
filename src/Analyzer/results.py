"""Structured, row-preserving results for model inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PredictionRow:
    """One input row and its prediction status."""

    row_index: Any
    inputs: Mapping[str, Any]
    prediction: np.ndarray | None = None
    status: str = "ok"
    error: str | None = None


@dataclass(frozen=True)
class PredictionResult(Sequence[PredictionRow]):
    """Prediction rows in the same order as the caller's input."""

    rows: tuple[PredictionRow, ...]
    targets: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def __iter__(self) -> Iterator[PredictionRow]:
        return iter(self.rows)

    @property
    def valid_rows(self) -> tuple[PredictionRow, ...]:
        return tuple(row for row in self.rows if row.status == "ok")

    @property
    def invalid_rows(self) -> tuple[PredictionRow, ...]:
        return tuple(row for row in self.rows if row.status != "ok")

    def legacy_dict(self, input_columns: Iterable[str]) -> dict[tuple[Any, ...], np.ndarray]:
        """Return the historical tuple-keyed mapping.

        Duplicate input tuples cannot be represented by the historical mapping. Callers
        that need every row should use this structured result or ``to_dataframe()``.
        """

        columns = tuple(input_columns)
        result = {}
        for row in self.valid_rows:
            result[tuple(row.inputs[column] for column in columns)] = row.prediction
        return result

    def to_dataframe(self):
        """Convert all rows, including invalid rows, to a pandas DataFrame."""

        import pandas as pd

        records = []
        for row in self.rows:
            record = {"row_index": row.row_index, **dict(row.inputs)}
            if row.prediction is None:
                for target in self.targets:
                    record[f"{target}_pred"] = np.nan
            else:
                values = np.asarray(row.prediction).reshape(-1)
                for index, target in enumerate(self.targets):
                    record[f"{target}_pred"] = values[index]
            record["prediction_status"] = row.status
            record["prediction_error"] = row.error
            records.append(record)
        return pd.DataFrame.from_records(records)

    def to_csv(self, path, **kwargs) -> str:
        """Write all rows atomically and return the resolved output path."""

        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        self.to_dataframe().to_csv(temporary, index=False, **kwargs)
        temporary.replace(destination)
        return str(destination)


@dataclass(frozen=True)
class UncertaintyResult:
    """Repeated stochastic predictions for one row-preserving input."""

    mean: PredictionResult
    std: PredictionResult
    samples: tuple[PredictionResult, ...]
    method: str
    seed: int | None

