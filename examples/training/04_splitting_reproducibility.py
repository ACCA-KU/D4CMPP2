"""Use reproducible scaffold splitting.

Change ``split_strategy`` to ``random`` for seeded row splitting. Use
``predefined`` only when the CSV has a ``set`` column containing train/val/test.
``auto`` selects predefined when that column exists and random otherwise.
"""

from D4CMPP2 import train


model_path = train(
    data="test",
    target=["Abs"],
    network="GCN",
    split_strategy="scaffold",
    random_seed=42,
    split_random_seed=42,
    deterministic_algorithms=True,
    device="cpu",
    max_epoch=2,
    batch_size=8,
)
print(f"Saved reproducible model: {model_path}")
print("Inspect split_report.json and split_assignments.csv in that directory.")

