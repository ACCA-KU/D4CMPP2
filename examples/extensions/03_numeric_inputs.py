"""Train a custom graph model that consumes a named numeric input.

The Dataset exposes a configured numeric column as ``<column>_var``. This
example uses ``temperature_var`` and predefined train/val/test labels.
"""

from pathlib import Path

import torch
import torch.nn as nn

import D4CMPP2
from D4CMPP2.networks.src.GCN import graph_sum_pool


class NumericGCN(D4CMPP2.MolecularNetwork):
    model_name = "numeric_gcn"
    input_contract = D4CMPP2.InputContract(
        required=(
            "compound_graphs",
            "compound_node_feature",
            "temperature_var",
        )
    )
    hyperparameters = {
        "hidden_dim": D4CMPP2.Hyperparameter(
            "int",
            default=32,
            low=1,
            search_low=16,
            search_high=128,
            step=16,
            grid=(16, 32, 64),
        ),
        "dropout": D4CMPP2.Hyperparameter(
            "float",
            default=0.1,
            low=0.0,
            high=0.5,
            search_low=0.0,
            search_high=0.5,
            grid=(0.0, 0.1, 0.2),
        ),
    }
    default_optimization_space = ("hidden_dim", "dropout")

    def __init__(self, config):
        super().__init__(config)
        self.node_projection = nn.Linear(
            self.config["node_dim"],
            self.config["hidden_dim"],
        )
        self.output = nn.Sequential(
            nn.Dropout(self.config["dropout"]),
            nn.Linear(self.config["hidden_dim"] + 1, self.config["target_dim"]),
        )

    def forward(self, **batch):
        self.validate_input(batch)
        node = torch.relu(
            self.node_projection(batch["compound_node_feature"])
        )
        molecular = graph_sum_pool(batch["compound_graphs"], node)
        temperature = batch["temperature_var"].float().reshape(-1, 1)
        return self.output(torch.cat((molecular, temperature), dim=1))


if __name__ == "__main__":
    D4CMPP2.register_network(NumericGCN, data_contract="molecule")
    data_path = (
        Path(__file__).resolve().parents[1] / "assets" / "tiny_numeric.csv"
    )
    model_path = D4CMPP2.train(
        data=str(data_path),
        target=["target"],
        network="numeric_gcn",
        numeric_input_columns=["temperature"],
        split_strategy="predefined",
        device="cpu",
        max_epoch=2,
        batch_size=4,
    )
    print(f"Saved numeric-input model: {model_path}")

    analyzer = D4CMPP2.Analyzer(
        model_path,
        device="cpu",
        save_result=False,
    )
    result = analyzer.predict_rows(
        compound=["CCO", "CCN"],
        temperature=[298.0, 310.0],
    )
    print(result.to_dataframe())

