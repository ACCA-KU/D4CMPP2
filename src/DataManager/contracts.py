"""Additive metadata for DataManager and Dataset batch contracts."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple


class BatchUnwrapper(Protocol):
    """Static typing aid for Dataset unwrapper callables."""

    def __call__(self, *args: Any, device: str = "cpu", **kwargs: Any) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True)
class DatasetBatchContract:
    """Describe keys returned by a built-in Dataset unwrapper."""

    name: str
    required_keys: Tuple[str, ...] = ("target",)
    optional_keys: Tuple[str, ...] = ()
    molecule_suffixes: Tuple[str, ...] = ()
    numeric_suffix: Optional[str] = None

    def expected_keys(
        self,
        molecule_columns: Sequence[str] = (),
        numeric_input_columns: Sequence[str] = (),
    ) -> Tuple[str, ...]:
        keys = list(self.required_keys)
        for column in molecule_columns:
            keys.extend(f"{column}{suffix}" for suffix in self.molecule_suffixes)
        if self.numeric_suffix is not None:
            keys.extend(
                f"{column}{self.numeric_suffix}"
                for column in numeric_input_columns
            )
        return tuple(dict.fromkeys(keys))


@dataclass(frozen=True)
class DataManagerContract:
    """Immutable snapshot available before NetworkManager construction."""

    manager: str
    graph_type: str
    dataset: str
    unwrapper: str
    feature_dimensions: Mapping[str, int]
    batch: DatasetBatchContract

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "feature_dimensions",
            MappingProxyType(dict(self.feature_dimensions)),
        )


GENERAL_BATCH_CONTRACT = DatasetBatchContract(
    name="generalized_graph",
    required_keys=("target",),
    optional_keys=("original_row_index",),
    molecule_suffixes=(
        "_graphs",
        "_node_feature",
        "_edge_feature",
        "_smiles",
    ),
    numeric_suffix="_var",
)

ISA_BATCH_CONTRACT = DatasetBatchContract(
    name="generalized_isa",
    required_keys=("target",),
    optional_keys=("original_row_index",),
    molecule_suffixes=(
        "_graphs",
        "_r_node",
        "_r2r_edge",
        "_i_node",
        "_i2i_edge",
        "_d_node",
        "_d2d_edge",
        "_smiles",
    ),
    numeric_suffix="_var",
)

LEGACY_GENERAL_BATCH_CONTRACT = DatasetBatchContract(
    name="legacy_general",
    required_keys=("graph", "node_feats", "edge_feats", "target", "smiles"),
)

LEGACY_SOLVENT_BATCH_CONTRACT = DatasetBatchContract(
    name="legacy_solvent",
    required_keys=(
        "graph",
        "node_feats",
        "edge_feats",
        "solv_graph",
        "solv_node_feats",
        "solv_edge_feats",
        "target",
        "smiles",
        "solv_smiles",
    ),
)

LEGACY_ISA_BATCH_CONTRACT = DatasetBatchContract(
    name="legacy_isa",
    required_keys=(
        "graph",
        "r_node",
        "r_edge",
        "i_node",
        "d_node",
        "d_edge",
        "target",
        "smiles",
    ),
)


def validate_data_manager_contract(
    manager: Any,
    config: Mapping[str, Any],
) -> Optional[DataManagerContract]:
    """Validate built-in metadata, or return None for metadata-free custom managers."""

    dataset = getattr(manager, "dataset", None)
    batch_contract = getattr(dataset, "batch_contract", None)
    if batch_contract is None:
        return None
    dataset_name = getattr(dataset, "__name__", type(dataset).__name__)
    if not isinstance(batch_contract, DatasetBatchContract):
        raise TypeError(
            f"Dataset {dataset_name!r} "
            "batch_contract must be DatasetBatchContract."
        )

    unwrapper = getattr(manager, "unwrapper", None)
    if not callable(unwrapper):
        raise TypeError(
            f"DataManager {type(manager).__name__!r} selected Dataset "
            f"{dataset_name!r} but its unwrapper is not callable."
        )

    dimension_keys = getattr(manager, "feature_dimension_keys", None)
    if dimension_keys is None:
        return None
    missing = [key for key in dimension_keys if key not in config]
    if missing:
        raise ValueError(
            f"DataManager {type(manager).__name__!r} did not initialize required "
            f"feature dimension keys {missing!r} before NetworkManager construction. "
            f"Available config keys: {sorted(config)!r}."
        )
    invalid = {
        key: config[key]
        for key in dimension_keys
        if isinstance(config[key], bool)
        or not isinstance(config[key], int)
        or config[key] < 0
    }
    if invalid:
        raise ValueError(
            f"DataManager {type(manager).__name__!r} produced invalid feature "
            f"dimensions {invalid!r}; expected non-negative integers."
        )

    return DataManagerContract(
        manager=type(manager).__name__,
        graph_type=str(getattr(manager, "graph_type", "unknown")),
        dataset=dataset_name,
        unwrapper=getattr(unwrapper, "__qualname__", repr(unwrapper)),
        feature_dimensions={key: config[key] for key in dimension_keys},
        batch=batch_contract,
    )
