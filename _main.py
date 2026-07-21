"""Compatibility alias for :mod:`D4CMPP2.src.api.training`."""

from importlib import import_module as _import_module
import sys as _sys


_sys.modules[__name__] = _import_module("D4CMPP2.src.api.training")
