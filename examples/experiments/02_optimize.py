"""Run resumable model-aware grid or Bayesian optimization.

Run:
    python examples/experiments/02_optimize.py grid
    python examples/experiments/02_optimize.py bayesian

Every trial performs a real training run. The deliberately small search below
is for API demonstration, not a recommended scientific search space.
"""

import argparse

from D4CMPP2 import optimize


parser = argparse.ArgumentParser()
parser.add_argument("strategy", choices=("grid", "bayesian"))
args = parser.parse_args()

if args.strategy == "grid":
    hp = {
        "hidden_dim": [32, 64],
        "dropout": [0.0, 0.2],
    }
    trial_options = {}
else:
    hp = {
        "hidden_dim": {"low": 32, "high": 64, "step": 16},
        "dropout": {"low": 0.0, "high": 0.3},
    }
    trial_options = {"n_trials": 4}

result = optimize(
    data="test",
    target=["Abs"],
    network="GCN",
    HP=hp,
    optimize_strategy=args.strategy,
    optimization_path=f"_Models/optimization_{args.strategy}",
    resume=True,
    random_seed=42,
    device="cpu",
    max_epoch=2,
    batch_size=8,
    **trial_options,
)
print("Best parameters:", result.best_params)
print("Best validation loss:", result.best_score)
print("Best model:", result.best_model_path)
print("Summary:", result.summary_path)

