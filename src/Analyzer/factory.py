"""Automatic saved-model Analyzer selection."""

from __future__ import annotations

import yaml

from .ISAAnalyzer import (
    ISAAnalyzer,
    ISAAnalyzer_v2,
    ISAAnalyzer_v1p3,
    mol_with_atom_index,
    showAtomHighlight,
)
from .ISAPNAnalyzer import (
    ISAPNAnalyzer,
    ISAPNAnalyzer_v2,
    ISAPNAnalyzer_v1p3,
    ISATPNAnalyzer,
    ISATPNAnalyzer_v2,
)
from .ISAwSAnalyzer import ISAwSAnalyzer
from .MolAnalyzer import MolAnalyzer, MolAnalyzer_v2, MolAnalyzer_v1p3
from .core import (
    InferenceCore,
    ModelArtifacts,
    predict_ensemble,
    resolve_model_artifacts,
)
from .interpretation import ISAAnalysisResult, ISAAnalysisRow
from .results import PredictionResult, PredictionRow, UncertaintyResult


def _version_tuple(value) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in str(value or "1.0").split("."))
    except ValueError as exc:
        raise ValueError(
            f"Saved model version {value!r} is not a dotted numeric version. "
            "Correct the 'version' value in config.yaml."
        ) from exc


def _is_isa_config(config: dict) -> bool:
    contract_fields = (
        "data_manager_module",
        "data_manager_class",
        "network_manager_module",
        "network_manager_class",
        "train_manager_module",
        "train_manager_class",
    )
    return any("isa" in str(config.get(field, "")).casefold() for field in contract_fields)


def _is_legacy_solvent_isa(config: dict) -> bool:
    molecule_columns = config.get("molecule_columns", ())
    if isinstance(molecule_columns, str):
        molecule_columns = (molecule_columns,)
    if any(str(column).casefold() == "solvent" for column in molecule_columns):
        return True

    identity_fields = (
        "network_id",
        "network",
        "data_manager_module",
        "data_manager_class",
    )
    return any(
        "withsolv" in str(config.get(field, "")).casefold()
        or str(config.get(field, "")).casefold().endswith("ws")
        for field in identity_fields
    )


def _is_isapn_config(config: dict) -> bool:
    """Recognize canonical ISATPN and legacy ISATPM saved identities."""

    identity_fields = ("network_id", "name", "network")
    identities = {
        str(config.get(field, "")).casefold().replace("_", "")
        for field in identity_fields
    }
    return any(
        identity in {"isatpn", "isatpnmodel", "isatpm", "isatpmmodel"}
        for identity in identities
    )


def _select_analyzer_class(config: dict):
    version = _version_tuple(config.get("version", "1.0"))
    if _is_isa_config(config):
        if _is_isapn_config(config):
            return ISAPNAnalyzer_v2 if version >= (1, 3) else ISAPNAnalyzer
        if version >= (1, 3):
            return ISAAnalyzer_v2
        if _is_legacy_solvent_isa(config):
            return ISAwSAnalyzer
        return ISAAnalyzer
    return MolAnalyzer_v2 if version >= (1, 3) else MolAnalyzer


def select_analyzer_class(model_path, *, model_dir=None):
    """Return the compatibility Analyzer class selected from saved config."""

    artifacts = resolve_model_artifacts(model_path, model_dir=model_dir)
    try:
        with artifacts.config.open(encoding="utf-8") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(
            f"Could not read saved Analyzer config {str(artifacts.config)!r}. "
            "Check that config.yaml is valid YAML."
        ) from exc
    if not isinstance(config, dict):
        raise ValueError(
            f"Saved Analyzer config {str(artifacts.config)!r} must contain a mapping, "
            f"got {type(config).__name__}."
        )
    return _select_analyzer_class(config)


def create_analyzer(model_path, save_result=False, **kwargs):
    """Construct the Analyzer implementation required by a saved model."""

    analyzer_class = select_analyzer_class(
        model_path,
        model_dir=kwargs.get("model_dir"),
    )
    return analyzer_class(model_path, save_result=save_result, **kwargs)


class Analyzer:
    """Callable facade that automatically selects a saved-model Analyzer.

    Existing ``Analyzer.MolAnalyzer(...)``-style calls remain available as
    class attributes for compatibility.
    """

    InferenceCore = InferenceCore
    ISAAnalyzer = ISAAnalyzer
    ISAAnalyzer_v2 = ISAAnalyzer_v2
    ISAAnalyzer_v1p3 = ISAAnalyzer_v1p3
    ISAPNAnalyzer = ISAPNAnalyzer
    ISAPNAnalyzer_v2 = ISAPNAnalyzer_v2
    ISAPNAnalyzer_v1p3 = ISAPNAnalyzer_v1p3
    ISATPNAnalyzer = ISATPNAnalyzer
    ISATPNAnalyzer_v2 = ISATPNAnalyzer_v2
    ISAAnalysisResult = ISAAnalysisResult
    ISAAnalysisRow = ISAAnalysisRow
    ISAwSAnalyzer = ISAwSAnalyzer
    ModelArtifacts = ModelArtifacts
    MolAnalyzer = MolAnalyzer
    MolAnalyzer_v2 = MolAnalyzer_v2
    MolAnalyzer_v1p3 = MolAnalyzer_v1p3
    PredictionResult = PredictionResult
    PredictionRow = PredictionRow
    UncertaintyResult = UncertaintyResult
    create_analyzer = staticmethod(create_analyzer)
    mol_with_atom_index = staticmethod(mol_with_atom_index)
    predict_ensemble = staticmethod(predict_ensemble)
    resolve_model_artifacts = staticmethod(resolve_model_artifacts)
    select_analyzer_class = staticmethod(select_analyzer_class)
    showAtomHighlight = staticmethod(showAtomHighlight)

    def __new__(cls, model_path, save_result=False, **kwargs):
        return create_analyzer(model_path, save_result=save_result, **kwargs)
