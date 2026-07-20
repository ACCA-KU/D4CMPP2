"""Predict two molecules with a newly saved model directory."""

import argparse

from D4CMPP2 import Analyzer


parser = argparse.ArgumentParser()
parser.add_argument("model_path")
parser.add_argument("--device", default="cpu")
parser.add_argument("--csv")
args = parser.parse_args()

analyzer = Analyzer(
    args.model_path,
    save_result=False,
    device=args.device,
)
if args.csv:
    print(analyzer.predict_csv(args.csv))
else:
    print(analyzer.predict_rows(["CCO", "CCN"]).to_dataframe())
