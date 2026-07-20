"""Select any of the 13 PyG model IDs from one command.

Examples:
    python examples/training/07_model_families.py GAT
    python examples/training/07_model_families.py MPNNwS
    python examples/training/07_model_families.py ISAT
"""

import argparse

from D4CMPP2 import train


GENERAL = ("GCN", "MPNN", "DMPNN", "AFP", "GAT")
SOLVENT = ("GCNwS", "MPNNwS", "DMPNNwS", "AFPwS", "GATwS")
ISA = ("ISAT", "ISATPN", "GC")

parser = argparse.ArgumentParser()
parser.add_argument("network", choices=GENERAL + SOLVENT + ISA)
parser.add_argument(
    "--explicit-h",
    action="store_true",
    help="Generate compound nodes with explicit hydrogen atoms.",
)
args = parser.parse_args()

kwargs = {
    "data": "test",
    "target": ["Abs"],
    "network": args.network,
    "device": "cpu",
    "max_epoch": 2,
    "batch_size": 4,
}
if args.network in SOLVENT:
    kwargs["molecule_columns"] = ["compound", "solvent"]
if args.network in ISA:
    kwargs["sculptor_index"] = (6, 2, 0)
if args.explicit_h:
    kwargs["explicit_h_columns"] = ["compound"]

print(train(**kwargs))
