"""Compare legacy mapping prediction with the row-preserving result.

Run:
    python examples/inference/01_prediction.py path/to/model
"""

import argparse

from D4CMPP2 import Analyzer


parser = argparse.ArgumentParser()
parser.add_argument("model_path")
args = parser.parse_args()

analyzer = Analyzer(args.model_path, device="cpu", save_result=False)

# Historical convenience form: duplicate SMILES cannot be represented twice.
print(analyzer.predict(["CCO", "CCN"]))

# Preferred form: duplicates, invalid rows, source indices, and errors survive.
rows = analyzer.predict_rows(
    compound=["CCO", "CCO", "not-a-smiles"],
)
print(rows.to_dataframe())

