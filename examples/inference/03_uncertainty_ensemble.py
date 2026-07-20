"""Estimate MC-dropout and independent-model ensemble dispersion.

Run:
    python examples/inference/03_uncertainty_ensemble.py MODEL_A MODEL_B

Both ensemble models must have compatible input columns and targets. Reported
standard deviations are model dispersion estimates, not calibrated intervals.
"""

import argparse

from D4CMPP2 import Analyzer


parser = argparse.ArgumentParser()
parser.add_argument("model_a")
parser.add_argument("model_b")
args = parser.parse_args()

first = Analyzer(args.model_a, device="cpu", save_result=False)
second = Analyzer(args.model_b, device="cpu", save_result=False)

mc = first.predict_uncertainty(
    compound=["CCO", "CCN"],
    samples=30,
    seed=42,
)
print("MC mean")
print(mc.mean.to_dataframe())
print("MC std")
print(mc.std.to_dataframe())

ensemble = Analyzer.predict_ensemble(
    [first, second],
    compound=["CCO", "CCN"],
)
print("Ensemble mean")
print(ensemble.mean.to_dataframe())
print("Ensemble std")
print(ensemble.std.to_dataframe())

