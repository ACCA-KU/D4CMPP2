"""Train two targets while preserving the package's per-target NaN mask.

``test.csv`` contains Abs and Emi; some Emi values are missing. A row may still
train the finite target, while a batch with no finite values at all is rejected.
"""

from D4CMPP2 import train


model_path = train(
    data="test",
    target=["Abs", "Emi"],
    network="GCN",
    device="cpu",
    max_epoch=2,
    batch_size=8,
    scaler="standard",
)
print(f"Saved multi-target model: {model_path}")

