"""Non-mutating data quality summaries for molecular CSV inputs."""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from rdkit import Chem


REPORT_SCHEMA_VERSION = 1


def _canonical_smiles(value):
    if pd.isna(value) or not str(value).strip():
        return None, "empty_smiles"
    molecule = Chem.MolFromSmiles(str(value))
    if molecule is None:
        return None, "invalid_smiles"
    return Chem.MolToSmiles(molecule, canonical=True), None


def _target_summary(frame, targets):
    result = {}
    for target in targets:
        if target not in frame:
            result[target] = {"missing_column": True}
            continue
        numeric = pd.to_numeric(frame[target], errors="coerce")
        usable = numeric.dropna()
        result[target] = {
            "count": int(len(numeric)),
            "usable_count": int(usable.size),
            "missing_count": int(numeric.isna().sum()),
            "min": float(usable.min()) if not usable.empty else None,
            "max": float(usable.max()) if not usable.empty else None,
            "mean": float(usable.mean()) if not usable.empty else None,
            "std": float(usable.std()) if usable.size > 1 else None,
        }
    return result


def build_data_quality_report(frame, config, graph_errors=None):
    """Build summary and issue rows without changing the input DataFrame."""
    molecule_columns = list(config.get("molecule_columns", ["compound"]))
    targets = list(config.get("target", []))
    issues = []
    canonical_columns = {}

    for column in molecule_columns:
        if column not in frame:
            issues.append({
                "issue": "missing_molecule_column",
                "row_index": None,
                "column": column,
                "value": None,
                "detail": "Configured molecule column is absent.",
            })
            continue
        canonical = []
        for row_index, value in frame[column].items():
            normalized, issue = _canonical_smiles(value)
            canonical.append(normalized)
            if issue:
                issues.append({
                    "issue": issue,
                    "row_index": row_index,
                    "column": column,
                    "value": None if pd.isna(value) else str(value),
                    "detail": "RDKit could not produce a canonical molecule." if issue == "invalid_smiles" else "SMILES is empty.",
                })
        canonical_columns[column] = pd.Series(canonical, index=frame.index)

    signature = None
    if len(canonical_columns) == len(molecule_columns):
        signature_frame = pd.DataFrame(canonical_columns)
        valid_signature = signature_frame.notna().all(axis=1)
        # pandas 3 may preserve missing values as float NaN after astype(str),
        # so convert every scalar explicitly before joining the row signature.
        signature = signature_frame.agg(
            lambda values: "||".join(str(value) for value in values),
            axis=1,
        ).where(valid_signature)
        duplicate_mask = signature.notna() & signature.duplicated(keep=False)
        for row_index in frame.index[duplicate_mask]:
            issues.append({
                "issue": "duplicate_molecule",
                "row_index": row_index,
                "column": "|".join(molecule_columns),
                "value": signature.loc[row_index],
                "detail": "Canonical molecule input appears in multiple rows.",
            })

        if "set" in frame:
            grouped = pd.DataFrame({"signature": signature, "set": frame["set"]}).dropna()
            overlap = grouped.groupby("signature")["set"].agg(lambda values: sorted(set(values)))
            for molecule, labels in overlap.items():
                if len(labels) > 1:
                    issues.append({
                        "issue": "split_molecule_overlap",
                        "row_index": None,
                        "column": "set",
                        "value": molecule,
                        "detail": f"Canonical molecule occurs across splits {labels}.",
                    })

    for target in targets:
        if target not in frame:
            continue
        numeric = pd.to_numeric(frame[target], errors="coerce")
        for row_index in frame.index[numeric.isna()]:
            issues.append({
                "issue": "missing_target",
                "row_index": row_index,
                "column": target,
                "value": None if pd.isna(frame.at[row_index, target]) else str(frame.at[row_index, target]),
                "detail": "Target is blank, NaN, or non-numeric.",
            })
        if signature is not None:
            grouped = pd.DataFrame({"signature": signature, "target": numeric}).dropna()
            conflicts = grouped.groupby("signature")["target"].nunique()
            for molecule in conflicts[conflicts > 1].index:
                issues.append({
                    "issue": "conflicting_duplicate_target",
                    "row_index": None,
                    "column": target,
                    "value": molecule,
                    "detail": "Duplicate canonical molecule has differing target values.",
                })

    for error in graph_errors or []:
        issues.append({
            "issue": "graph_generation_failure",
            "row_index": error.get("row_index"),
            "column": error.get("type"),
            "value": str(error.get("smiles")),
            "detail": str(error.get("reason")),
        })

    issue_counts = {}
    for issue in issues:
        issue_counts[issue["issue"]] = issue_counts.get(issue["issue"], 0) + 1
    split_counts = (
        {str(key): int(value) for key, value in frame["set"].value_counts(dropna=False).items()}
        if "set" in frame else {"automatic": int(len(frame))}
    )
    report = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_path": str(config.get("DATA_PATH", "")),
        "row_count": int(len(frame)),
        "molecule_columns": molecule_columns,
        "target_columns": targets,
        "split_counts": split_counts,
        "target_summary": _target_summary(frame, targets),
        "issue_counts": issue_counts,
        "issue_count": len(issues),
    }
    return report, pd.DataFrame(
        issues,
        columns=["issue", "row_index", "column", "value", "detail"],
    )


def write_data_quality_report(frame, config, graph_errors=None):
    """Atomically write JSON summary and issue CSV; return both paths."""
    model_path = Path(config["MODEL_PATH"])
    model_path.mkdir(parents=True, exist_ok=True)
    report, issues = build_data_quality_report(frame, config, graph_errors)
    json_path = model_path / "data_quality_report.json"
    csv_path = model_path / "data_quality_issues.csv"
    token = uuid.uuid4().hex
    json_staging = json_path.with_name(f".{json_path.name}.{token}.tmp")
    csv_staging = csv_path.with_name(f".{csv_path.name}.{token}.tmp")
    try:
        json_staging.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        issues.to_csv(csv_staging, index=False)
        os.replace(json_staging, json_path)
        os.replace(csv_staging, csv_path)
    finally:
        for staging in (json_staging, csv_staging):
            if staging.exists():
                staging.unlink()
    return json_path, csv_path
