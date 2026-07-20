"""Train an interpretable ISA-family model.

Choose ISAT for positive attention, ISATPN for positive/negative fragment
attention and hidden features, or GC for the compatible group-contribution
family. ``sculptor_index`` is saved and must be reused during inference.
"""

from D4CMPP2 import train


model_path = train(
    data="Aqsoldb",
    target=["Solubility"],
    network="ISATPN",
    sculptor_index=(6, 2, 0),
    device="cpu",
    max_epoch=2,
    batch_size=16,
)
print(f"Saved ISA model: {model_path}")
print("The model directory includes the training-time functional_group.csv.")

