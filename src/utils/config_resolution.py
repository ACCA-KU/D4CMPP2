"""Immutable, provenance-aware helpers for compatibility config merging."""

import copy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


RUNTIME_CONFIG_KEYS = (
    "TRANSFER_PATH",
    "RESUME_PATH",
    "loaded",
    "full_resume",
)


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime-only mode values removed from the pipeline working config."""

    transfer_path: Optional[str] = None
    resume_path: Optional[str] = None
    loaded: bool = False
    full_resume: bool = False


@dataclass(frozen=True)
class ConfigResolution:
    """Immutable resolved config snapshot and key-level source provenance."""

    values: Mapping[str, Any]
    provenance: Mapping[str, str]

    @classmethod
    def from_working(
        cls,
        values: Mapping[str, Any],
        provenance: Mapping[str, str],
    ) -> "ConfigResolution":
        copied_values = copy.deepcopy(dict(values))
        copied_provenance = dict(provenance)
        return cls(
            MappingProxyType(copied_values),
            MappingProxyType(copied_provenance),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return an isolated mutable working copy."""

        return copy.deepcopy(dict(self.values))


def merge_config_layers(
    layers: Iterable[Tuple[str, Mapping[str, Any]]],
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Deep-copy and merge named layers from lowest to highest precedence."""

    result: Dict[str, Any] = {}
    provenance: Dict[str, str] = {}
    for source, layer in layers:
        if not isinstance(layer, Mapping):
            raise TypeError(
                f"Config layer {source!r} must be a mapping, got "
                f"{type(layer).__name__}."
            )
        for key, value in layer.items():
            result[key] = copy.deepcopy(value)
            provenance[key] = source
    return result, provenance


def overlay_config_layer(
    values: Mapping[str, Any],
    provenance: Mapping[str, str],
    layer: Mapping[str, Any],
    source: Optional[str] = None,
    layer_provenance: Optional[Mapping[str, str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Overlay one layer while retaining provenance for untouched keys."""

    if not isinstance(layer, Mapping):
        raise TypeError(
            f"Config overlay must be a mapping, got {type(layer).__name__}."
        )
    result = copy.deepcopy(dict(values))
    result_provenance = dict(provenance)
    for key, value in layer.items():
        result[key] = copy.deepcopy(value)
        if layer_provenance is not None and key in layer_provenance:
            result_provenance[key] = layer_provenance[key]
        elif source is not None:
            result_provenance[key] = source
        else:
            raise ValueError(
                f"Config overlay key {key!r} has no provenance source."
            )
    return result, result_provenance


def mark_derived(
    provenance: Dict[str, str],
    values: Mapping[str, Any],
    source: str = "derived",
) -> None:
    """Assign provenance to keys introduced by path/data normalization."""

    for key in values:
        provenance.setdefault(key, source)


def split_runtime_config(
    config: Mapping[str, Any],
) -> Tuple[Dict[str, Any], RuntimeConfig]:
    """Return an isolated pipeline config and runtime mode without input mutation."""

    working = copy.deepcopy(dict(config))
    runtime = RuntimeConfig(
        transfer_path=working.pop("TRANSFER_PATH", None),
        resume_path=working.pop("RESUME_PATH", None),
        loaded=bool(working.pop("loaded", False)),
        full_resume=bool(working.pop("full_resume", False)),
    )
    return working, runtime
