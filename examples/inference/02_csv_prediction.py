"""Run atomic row-preserving CSV inference with optional MC-dropout columns.

Run:
    python examples/inference/02_csv_prediction.py MODEL INPUT.csv OUTPUT.csv

INPUT.csv must contain every molecule and numeric input column saved in the
model config. Invalid input rows remain in OUTPUT.csv with an error message.
"""

import argparse

from D4CMPP2 import Analyzer


parser = argparse.ArgumentParser()
parser.add_argument("model_path")
parser.add_argument("input_csv")
parser.add_argument("output_csv")
parser.add_argument("--uncertainty-samples", type=int)
args = parser.parse_args()

analyzer = Analyzer(args.model_path, device="cpu", save_result=False)
output = analyzer.predict_csv(
    args.input_csv,
    args.output_csv,
    uncertainty_samples=args.uncertainty_samples,
    uncertainty_seed=42,
)
print(f"Prediction CSV: {output}")

