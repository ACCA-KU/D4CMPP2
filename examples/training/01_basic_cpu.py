"""Train a small GCN on CPU and print the committed model directory.

Run:
    python examples/training/01_basic_cpu.py

The bundled ``test.csv`` is used so this also serves as an installation smoke
test. Real runs should replace ``data`` and ``target``.
"""

from D4CMPP2 import train


model_path = train(
    data="test",
    target=["Abs"],
    network="GCN",
    device="cpu",
    max_epoch=2,
    batch_size=4,
    random_seed=42,
    split_random_seed=42,
)
print(f"Saved model: {model_path}")

