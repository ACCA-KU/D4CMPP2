"""Lazy exports for built-in training-manager modules."""

from importlib import import_module
from types import ModuleType

__all__ = ["TrainManager", "ISATrainManager", "callbacks"]


def __getattr__(name: str) -> ModuleType:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module
