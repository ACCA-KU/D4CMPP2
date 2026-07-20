"""Read aligned ISA scores and optional ISATPN hidden features.

Run:
    python examples/inference/04_isa_interpretation.py path/to/ISA/model
"""

import argparse

from D4CMPP2 import Analyzer


parser = argparse.ArgumentParser()
parser.add_argument("model_path")
parser.add_argument("--smiles", default="CCOC(=O)c1ccccc1")
args = parser.parse_args()

analyzer = Analyzer(args.model_path, device="cpu", save_result=False)
include_features = "PN" in type(analyzer).__name__
analysis = analyzer.analyze_rows(
    [args.smiles],
    include_features=include_features,
)
row = analysis[0]

print("Fragments:", row.fragment_atom_indices)
print("Score mode:", row.score_mode)
print("Positive score:", row.scores["positive"])
print("Positive atom score:", row.atom_scores("positive"))
if "negative" in row.scores:
    print("Negative score:", row.scores["negative"])
if row.features:
    print("Feature keys:", sorted(row.features))

# For an interactive plot:
# analyzer.plot_analysis(row, score="positive")  # ISAT/GC
# analyzer.plot_analysis(row)                    # ISATPN

