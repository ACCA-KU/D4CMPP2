"""Build a target-aware leaderboard from model directories.

Run:
    python examples/experiments/01_compare.py _Models --metric val_rmse
"""

import argparse

from D4CMPP2 import compare_experiments


parser = argparse.ArgumentParser()
parser.add_argument("roots", nargs="+")
parser.add_argument("--metric", default="val_rmse")
parser.add_argument("--target")
parser.add_argument("--output", default="leaderboard.csv")
args = parser.parse_args()

leaderboard = compare_experiments(
    args.roots,
    output_path=args.output,
    metric=args.metric,
    target=args.target,
)
print(leaderboard)

