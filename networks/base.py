"""Small, explicit contracts shared by D4CMPP2 molecular networks."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar, Mapping

import torch
import torch.nn as nn


_MISSING = object()


@dataclass(frozen=True)
class Hyperparameter:
    """One validated model field and its optional optimization domain."""

    kind: str
    default: object = _MISSING
    low: float | int | None = None
    high: float | int | None = None
    search_low: float | int | None = None
    search_high: float | int | None = None
    step: float | int | None = None
    values: tuple[object, ...] = ()
    scale: str = "linear"
    grid: tuple[object, ...] = ()
    description: str = ""

    def validate(self, name, value):
        if self.kind == "int":
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer, got {value!r}.")
        elif self.kind == "float":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number, got {value!r}.")
            value = float(value)
        elif self.kind == "categorical":
            if value not in self.values:
                raise ValueError(
                    f"{name} must be one of {list(self.values)!r}, got {value!r}."
                )
            return value
        else:
            raise ValueError(
                f"Hyperparameter {name!r} has unknown kind {self.kind!r}."
            )
        if self.low is not None and value < self.low:
            raise ValueError(f"{name} must be >= {self.low}, got {value!r}.")
        if self.high is not None and value > self.high:
            raise ValueError(f"{name} must be <= {self.high}, got {value!r}.")
        return value


STANDARD_GRAPH_HYPERPARAMETERS = MappingProxyType(
    {
        "hidden_dim": Hyperparameter(
            "int", default=64, low=1, search_low=16, search_high=256,
            step=16, grid=(32, 64, 128, 256),
            description="Hidden graph representation width.",
        ),
        "conv_layers": Hyperparameter(
            "int", default=4, low=1, search_low=1, search_high=8,
            step=1, grid=(2, 4, 6, 8),
            description="Number of message-passing layers.",
        ),
        "linear_layers": Hyperparameter(
            "int", default=2, low=1, search_low=1, search_high=5,
            step=1, grid=(1, 2, 3, 4),
            description="Number of prediction-head linear layers.",
        ),
        "dropout": Hyperparameter(
            "float", default=0.2, low=0.0, high=0.5,
            search_low=0.0, search_high=0.5,
            grid=(0.0, 0.1, 0.2, 0.3, 0.5),
            description="Dropout probability.",
        ),
    }
)
STANDARD_GRAPH_OPTIMIZATION_SPACE = tuple(STANDARD_GRAPH_HYPERPARAMETERS)
STANDARD_SOLVENT_HYPERPARAMETERS = MappingProxyType(
    {
        **STANDARD_GRAPH_HYPERPARAMETERS,
        "solvent_hidden_dim": Hyperparameter(
            "int", default=64, low=1, search_low=16, search_high=256,
            step=16, grid=(32, 64, 128),
            description="Hidden width of the solvent graph branch.",
        ),
        "solvent_conv_layers": Hyperparameter(
            "int", default=4, low=1, search_low=1, search_high=8,
            step=1, grid=(2, 4, 6),
            description="Number of solvent graph convolution layers.",
        ),
        "solvent_linear_layers": Hyperparameter(
            "int", default=2, low=1, search_low=1, search_high=5,
            step=1, grid=(1, 2, 3),
            description="Number of solvent projection layers.",
        ),
        "solvent_dropout": Hyperparameter(
            "float", default=0.2, low=0.0, high=0.5,
            search_low=0.0, search_high=0.5,
            grid=(0.0, 0.1, 0.2, 0.3, 0.5),
            description="Dropout probability in the solvent branch.",
        ),
    }
)
STANDARD_SOLVENT_OPTIMIZATION_SPACE = tuple(STANDARD_SOLVENT_HYPERPARAMETERS)
STANDARD_SOLVENT_CONFIG_ALIASES = MappingProxyType(
    {
        "solv_hidden_dim": "solvent_hidden_dim",
        "solv_conv_layers": "solvent_conv_layers",
        "solv_linear_layers": "solvent_linear_layers",
        "solv_dropout": "solvent_dropout",
    }
)
ISA_HYPERPARAMETERS = MappingProxyType(
    {
        **STANDARD_GRAPH_HYPERPARAMETERS,
        "linear_layers": Hyperparameter(
            "int", default=4, low=1, high=4,
            search_low=1, search_high=4, step=1, grid=(1, 2, 3, 4),
            description="Number of prediction-head linear layers.",
        ),
        "dropout": Hyperparameter(
            "float", default=0.1, low=0.0, high=0.5,
            search_low=0.0, search_high=0.5,
            grid=(0.0, 0.1, 0.2, 0.3, 0.5),
            description="Dropout probability.",
        ),
    }
)
ISA_OPTIMIZATION_SPACE = tuple(ISA_HYPERPARAMETERS)


class ModelConfig(Mapping):
    """Immutable, model-only view of the larger training configuration."""

    def __init__(self, values):
        self._values = MappingProxyType(dict(values))

    def __getitem__(self, key):
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __repr__(self):
        return f"ModelConfig({dict(self._values)!r})"


@dataclass(frozen=True)
class InputContract:
    required: tuple[str, ...]
    optional: tuple[str, ...] = ()

    def validate(self, batch):
        missing = [name for name in self.required if name not in batch]
        if missing:
            raise ValueError(
                f"Network input is missing required fields {missing!r}. "
                f"Required fields: {list(self.required)!r}."
            )


def masked_mse(prediction, target):
    mask = ~torch.isnan(target)
    if not torch.any(mask):
        raise ValueError("A batch contains no finite target values for loss.")
    return torch.mean((prediction[mask] - target[mask]) ** 2)


class MolecularNetwork(nn.Module, ABC):
    """Base contract for trainable molecular property networks."""

    model_name: ClassVar[str]
    input_contract: ClassVar[InputContract]
    required_config: ClassVar[tuple[str, ...]] = (
        "node_dim",
        "target_dim",
    )
    hyperparameters: ClassVar[Mapping[str, Hyperparameter]] = {}
    default_optimization_space: ClassVar[tuple[str, ...]] = ()
    config_aliases: ClassVar[Mapping[str, str]] = STANDARD_SOLVENT_CONFIG_ALIASES

    def __init__(self, config):
        super().__init__()
        self.config = self.validate_config(config)

    @classmethod
    def validate_config(cls, config):
        normalized = dict(config)
        for old_name, canonical_name in cls.config_aliases.items():
            if old_name in normalized:
                if (
                    canonical_name in normalized
                    and normalized[canonical_name] != normalized[old_name]
                ):
                    raise ValueError(
                        f"{cls.__name__} received conflicting values for legacy "
                        f"{old_name!r} and canonical {canonical_name!r}."
                    )
                normalized.setdefault(canonical_name, normalized[old_name])
        missing = [name for name in cls.required_config if name not in normalized]
        if missing:
            raise ValueError(
                f"{cls.__name__} config is missing required fields {missing!r}."
            )
        values = {name: normalized[name] for name in cls.required_config}
        for name, field in cls.hyperparameters.items():
            if name in normalized and normalized[name] is not None:
                value = normalized[name]
            elif field.default is not _MISSING:
                value = field.default
            else:
                raise ValueError(
                    f"{cls.__name__} config requires hyperparameter {name!r}."
                )
            values[name] = field.validate(name, value)
        return ModelConfig(values)

    @classmethod
    def optimization_space(cls, keys=None):
        selected = (
            cls.default_optimization_space if keys is None else tuple(keys)
        )
        unknown = [name for name in selected if name not in cls.hyperparameters]
        if unknown:
            raise ValueError(
                f"{cls.model_name} does not define hyperparameters {unknown!r}. "
                f"Available keys: {sorted(cls.hyperparameters)!r}."
            )
        return {name: cls.hyperparameters[name] for name in selected}

    def validate_input(self, batch):
        self.input_contract.validate(batch)

    def compute_loss(self, prediction, target):
        return masked_mse(prediction, target)

    def loss_fn(self, prediction, target):
        """Temporary training-manager bridge during the staged migration."""
        return self.compute_loss(prediction, target)

    @abstractmethod
    def forward(self, **batch):
        raise NotImplementedError
