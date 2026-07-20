import os
import unittest


HEAVY_ENV_VAR = "D4CMPP2_RUN_HEAVY"


def heavy_test(test):
    """Mark a PyG/RDKit/model test and skip it unless explicitly enabled."""
    enabled = os.environ.get(HEAVY_ENV_VAR, "").strip().lower() in {"1", "true", "yes", "on"}
    return unittest.skipUnless(
        enabled,
        f"heavy test; set {HEAVY_ENV_VAR}=1 in an environment with ML dependencies",
    )(test)
