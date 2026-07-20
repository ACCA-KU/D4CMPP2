"""Train a model with separate compound and solvent graph branches.

The bundled test data contains ``compound``, ``solvent`` and ``Abs`` columns.
Other solvent-aware IDs are MPNNwS, DMPNNwS, AFPwS, and GATwS.
"""

from D4CMPP2 import train


model_path = train(
    data="test",
    target=["Abs"],
    network="GCNwS",
    molecule_columns=["compound", "solvent"],
    device="cpu",
    max_epoch=2,
    batch_size=4,
)
print(f"Saved solvent model: {model_path}")

