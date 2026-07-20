"""Use the compatibility grid_search API.

New projects should prefer D4CMPP2.optimize. This function retains the
historical ``None`` return value and continues after an individual trial error.
"""

from D4CMPP2 import grid_search


result = grid_search(
    {
        "hidden_dim": [32, 64],
        "dropout": [0.0, 0.2],
    },
    data="test",
    target=["Abs"],
    network="GCN",
    device="cpu",
    max_epoch=2,
    batch_size=8,
)
print("Historical return value:", result)

