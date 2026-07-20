"""Dependency-light CSV schema and split validation helpers."""

import math


ALLOWED_SET_LABELS = ("train", "val", "test")


def validate_csv_schema(path, columns, row_count, molecule_columns, target_columns, numeric_columns):
    """Validate required columns and basic table shape without mutating data."""
    if row_count == 0:
        raise ValueError(
            f"CSV file {path!r} contains no data rows. Add at least one row and the required columns."
        )

    groups = {
        "molecule": list(molecule_columns),
        "target": list(target_columns),
        "numeric input": list(numeric_columns),
    }
    duplicates = {
        name: sorted({column for column in requested if requested.count(column) > 1})
        for name, requested in groups.items()
    }
    duplicates = {name: values for name, values in duplicates.items() if values}
    if duplicates:
        raise ValueError(
            f"Duplicate configured CSV columns were found for {path!r}: {duplicates}. "
            "List each molecule, target, and numeric input column once."
        )

    available = list(columns)
    missing = {
        name: [column for column in requested if column not in available]
        for name, requested in groups.items()
    }
    missing = {name: values for name, values in missing.items() if values}
    if missing:
        raise ValueError(
            f"Required CSV columns were not found in {path!r}: {missing}. "
            f"Available columns: {available}. Check molecule_columns, target, and "
            "numeric_input_columns spelling."
        )


def validate_numeric_columns(path, invalid_rows_by_column, kind):
    """Report non-numeric cells with original row indices."""
    invalid = {column: list(rows) for column, rows in invalid_rows_by_column.items() if rows}
    if invalid:
        raise ValueError(
            f"{kind} columns in {path!r} contain non-numeric values at row indices: {invalid}. "
            "Replace those values with numbers or blank/NaN values."
        )


def validate_nonempty_targets(path, all_nan_by_column):
    """Reject target columns that contain no usable values."""
    empty = [column for column, all_nan in all_nan_by_column.items() if all_nan]
    if empty:
        raise ValueError(
            f"Target columns {empty} in {path!r} contain only NaN/blank values. "
            "Provide at least one numeric target value in each target column."
        )


def validate_set_labels(path, values, require_train_val=True):
    """Validate explicit split labels and return their counts."""
    labels = list(values)
    invalid = sorted({repr(value) for value in labels if _is_missing(value) or value not in ALLOWED_SET_LABELS})
    counts = {label: labels.count(label) for label in ALLOWED_SET_LABELS}
    if invalid:
        raise ValueError(
            f"Column 'set' in {path!r} contains invalid labels {invalid}. "
            f"Allowed labels: {list(ALLOWED_SET_LABELS)}. Split counts: {counts}."
        )
    if require_train_val and (counts["train"] == 0 or counts["val"] == 0):
        missing = [label for label in ("train", "val") if counts[label] == 0]
        raise ValueError(
            f"Column 'set' in {path!r} is missing required split labels {missing}. "
            f"Split counts: {counts}. Add at least one train and one val row."
        )
    return counts


def validate_automatic_split_size(path, row_count, minimum=10):
    """Reject datasets too small for the legacy 80/10/10 automatic split."""
    if row_count < minimum:
        raise ValueError(
            f"Dataset {path!r} has {row_count} usable rows, but automatic splitting requires "
            f"at least {minimum}. Add rows or provide a 'set' column with train/val labels."
        )


def validate_aligned_lengths(path, lengths):
    """Assert that extracted CSV arrays still share one row count."""
    normalized = {name: int(length) for name, length in lengths.items()}
    if len(set(normalized.values())) > 1:
        raise ValueError(
            f"CSV-derived arrays from {path!r} have inconsistent lengths: {normalized}. "
            "Check that all molecule, numeric input, target, and set values use the same rows."
        )


def _is_missing(value):
    return value is None or (isinstance(value, float) and math.isnan(value))
