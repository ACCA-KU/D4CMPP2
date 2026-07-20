"""Register and train the complete custom model in examples/custom_network.py."""

import sys
from pathlib import Path

from D4CMPP2 import train


# Make the sibling example module importable when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from custom_network import register  # noqa: E402


definition = register()
print(f"Registered: {definition.network.model_name}")

model_path = train(
    data="test",
    target=["Abs"],
    network="custom_gcn",
    hidden_dim=32,
    dropout=0.1,
    huber_delta=1.0,
    device="cpu",
    max_epoch=2,
    batch_size=8,
)
print(f"Saved custom model: {model_path}")
print("Its source snapshot is stored as network.py in the model directory.")

