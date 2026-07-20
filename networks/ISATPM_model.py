"""Deprecated compatibility wrapper for the former ISATPM module name."""

import warnings

warnings.warn(
    "D4CMPP2.networks.ISATPM_model is deprecated; use "
    "D4CMPP2.networks.ISATPN_model instead.",
    FutureWarning,
    stacklevel=2,
)

from D4CMPP2.networks.ISATPN_model import ISATPN, network

__all__ = ["ISATPN", "network"]
