"""Resume, continue, or transfer from a saved model.

Run one mode at a time:
    python examples/training/06_saved_model_modes.py resume path/to/model
    python examples/training/06_saved_model_modes.py continue path/to/model
    python examples/training/06_saved_model_modes.py transfer path/to/model
"""

import argparse

from D4CMPP2 import train


parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=("resume", "continue", "transfer"))
parser.add_argument("model_path")
args = parser.parse_args()

if args.mode == "resume":
    # Exact state: optimizer, schedulers, epoch, early stopping, and RNG.
    result = train(
        RESUME_PATH=args.model_path,
        device="cpu",
        max_epoch=4,
    )
elif args.mode == "continue":
    # final.pth is loaded, but optimizer, schedulers, epoch, and RNG restart.
    result = train(
        LOAD_PATH=args.model_path,
        device="cpu",
        max_epoch=2,
    )
else:
    # A new model/data run receives only compatible same-name/same-shape weights.
    result = train(
        TRANSFER_PATH=args.model_path,
        data="test",
        target=["Abs"],
        network="GCN",
        device="cpu",
        max_epoch=2,
        lr_dict={"GCNs": 1e-4},
    )

print(f"Result model: {result}")
