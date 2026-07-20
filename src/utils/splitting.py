"""Molecular split strategies and auditable split reports."""

import csv
import json
import os
import random
import uuid
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


SPLIT_REPORT_SCHEMA_VERSION = 1
SPLIT_NAMES = ("train", "val", "test")
DEFAULT_SPLIT_FRACTIONS = (0.8, 0.1, 0.1)


def murcko_scaffold(smiles, include_chirality=False):
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(
            f"RDKit could not parse SMILES {smiles!r} while computing a Murcko scaffold. "
            "Fix the molecule or ensure invalid graph rows are filtered before splitting."
        )
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    return Chem.MolToSmiles(
        scaffold,
        canonical=True,
        isomericSmiles=bool(include_chirality),
    )


def scaffold_split_indices(
    smiles,
    seed=42,
    fractions=DEFAULT_SPLIT_FRACTIONS,
    include_chirality=False,
):
    """Assign whole Murcko-scaffold groups to reproducible 80/10/10 splits."""
    fractions = tuple(float(value) for value in fractions)
    if len(fractions) != 3 or any(value <= 0 for value in fractions):
        raise ValueError(
            f"split fractions must contain three positive values, got {fractions!r}."
        )
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(
            f"split fractions must sum to 1.0, got {fractions!r} "
            f"(sum={sum(fractions)!r})."
        )
    groups = {}
    row_scaffolds = []
    for index, value in enumerate(smiles):
        scaffold = murcko_scaffold(value, include_chirality=include_chirality)
        row_scaffolds.append(scaffold)
        groups.setdefault(scaffold, []).append(index)
    if len(groups) < 3:
        raise ValueError(
            f"Scaffold splitting requires at least 3 distinct Murcko scaffolds, "
            f"but found {len(groups)} across {len(row_scaffolds)} usable rows. "
            "Add structurally diverse molecules or use split_strategy='random' or "
            "a predefined 'set' column."
        )

    rng = random.Random(seed)
    grouped = list(groups.items())
    rng.shuffle(grouped)
    grouped.sort(key=lambda item: -len(item[1]))
    target_counts = np.asarray(fractions) * len(row_scaffolds)
    assigned_groups = {name: [] for name in SPLIT_NAMES}
    assigned_indices = {name: [] for name in SPLIT_NAMES}

    for scaffold, indices in grouped:
        scores = []
        for position, name in enumerate(SPLIT_NAMES):
            projected = np.asarray(
                [len(assigned_indices[item]) for item in SPLIT_NAMES],
                dtype=float,
            )
            projected[position] += len(indices)
            scores.append(float(np.abs(projected - target_counts).sum()))
        selected = SPLIT_NAMES[min(range(3), key=lambda index: (scores[index], index))]
        assigned_groups[selected].append(scaffold)
        assigned_indices[selected].extend(indices)

    for empty_name in [
        name for name in SPLIT_NAMES if not assigned_indices[name]
    ]:
        donors = [
            name for name in SPLIT_NAMES if len(assigned_groups[name]) > 1
        ]
        if not donors:
            raise ValueError(
                "Scaffold grouping could not produce non-empty train, val, and test "
                "splits without breaking a scaffold group. Use a more diverse dataset."
            )
        donor = max(donors, key=lambda name: len(assigned_indices[name]))
        scaffold = min(
            assigned_groups[donor],
            key=lambda value: (len(groups[value]), value),
        )
        assigned_groups[donor].remove(scaffold)
        assigned_groups[empty_name].append(scaffold)
        moved = groups[scaffold]
        assigned_indices[donor] = [
            index for index in assigned_indices[donor] if index not in set(moved)
        ]
        assigned_indices[empty_name].extend(moved)

    result = tuple(
        np.asarray(sorted(assigned_indices[name]), dtype=int)
        for name in SPLIT_NAMES
    )
    scaffold_sets = [
        {row_scaffolds[index] for index in indices} for indices in result
    ]
    overlap = (
        scaffold_sets[0] & scaffold_sets[1]
        | scaffold_sets[0] & scaffold_sets[2]
        | scaffold_sets[1] & scaffold_sets[2]
    )
    if overlap:
        raise RuntimeError(
            f"Internal scaffold split error: scaffolds crossed split boundaries: "
            f"{sorted(overlap)!r}."
        )
    return result, row_scaffolds


def _target_stats(values):
    array = np.asarray(values, dtype=float)
    result = []
    for column in range(array.shape[1]):
        usable = array[:, column]
        usable = usable[~np.isnan(usable)]
        result.append(
            {
                "count": int(array.shape[0]),
                "usable_count": int(usable.size),
                "missing_count": int(array.shape[0] - usable.size),
                "min": float(usable.min()) if usable.size else None,
                "max": float(usable.max()) if usable.size else None,
                "mean": float(usable.mean()) if usable.size else None,
                "std": float(usable.std(ddof=1)) if usable.size > 1 else None,
            }
        )
    return result


def write_split_report(
    config,
    strategy,
    indices_by_split,
    row_indices,
    target_values,
    target_names,
    smiles,
    scaffold_column,
    include_chirality=False,
):
    """Write atomic JSON summary and row-level CSV split assignments."""
    scaffolds = [
        murcko_scaffold(value, include_chirality=include_chirality)
        for value in smiles
    ]
    scaffold_sets = {
        name: {scaffolds[index] for index in indices}
        for name, indices in zip(SPLIT_NAMES, indices_by_split)
    }
    overlap = sorted(
        scaffold_sets["train"] & scaffold_sets["val"]
        | scaffold_sets["train"] & scaffold_sets["test"]
        | scaffold_sets["val"] & scaffold_sets["test"]
    )
    if strategy == "scaffold" and overlap:
        raise RuntimeError(
            f"Scaffold leakage detected after scaffold splitting: {overlap!r}."
        )

    split_target_summary = {}
    for name, indices in zip(SPLIT_NAMES, indices_by_split):
        stats = _target_stats(np.asarray(target_values)[indices])
        split_target_summary[name] = {
            target: value for target, value in zip(target_names, stats)
        }
    report = {
        "split_report_schema_version": SPLIT_REPORT_SCHEMA_VERSION,
        "strategy": strategy,
        "split_random_seed": config.get("split_random_seed", 42),
        "fractions": dict(zip(SPLIT_NAMES, DEFAULT_SPLIT_FRACTIONS)),
        "scaffold_column": scaffold_column,
        "scaffold_include_chirality": bool(include_chirality),
        "counts": {
            name: int(len(indices))
            for name, indices in zip(SPLIT_NAMES, indices_by_split)
        },
        "scaffold_counts": {
            name: len(scaffold_sets[name]) for name in SPLIT_NAMES
        },
        "scaffold_overlap_count": len(overlap),
        "scaffold_overlap": overlap,
        "target_summary": split_target_summary,
    }

    model_path = Path(config["MODEL_PATH"])
    model_path.mkdir(parents=True, exist_ok=True)
    json_path = model_path / "split_report.json"
    csv_path = model_path / "split_assignments.csv"
    token = uuid.uuid4().hex
    json_staging = json_path.with_name(f".{json_path.name}.{token}.tmp")
    csv_staging = csv_path.with_name(f".{csv_path.name}.{token}.tmp")
    try:
        json_staging.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        assignment = {}
        for name, indices in zip(SPLIT_NAMES, indices_by_split):
            for index in indices:
                assignment[int(index)] = name
        with open(csv_staging, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "dataset_index",
                    "original_row_index",
                    "split",
                    "scaffold",
                    "smiles",
                ],
            )
            writer.writeheader()
            for index in range(len(smiles)):
                writer.writerow(
                    {
                        "dataset_index": index,
                        "original_row_index": int(row_indices[index]),
                        "split": assignment[index],
                        "scaffold": scaffolds[index],
                        "smiles": smiles[index],
                    }
                )
        os.replace(json_staging, json_path)
        os.replace(csv_staging, csv_path)
    finally:
        for staging in (json_staging, csv_staging):
            if staging.exists():
                staging.unlink()
    return report
