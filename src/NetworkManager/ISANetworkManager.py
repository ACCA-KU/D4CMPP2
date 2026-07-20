import torch
import importlib
import os
import yaml
import pandas as pd
from D4CMPP2.src.utils import PATH
from .NetworkManager import NetworkManager
    
class ISANetworkManager(NetworkManager):
    """Compatibility extension point referenced by ISA registries and saved configs."""

    pass
