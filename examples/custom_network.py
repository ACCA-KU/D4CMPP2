"""Minimal custom network with its own search space and loss function."""

import torch
import torch.nn as nn

import D4CMPP2
from D4CMPP2.networks.src.GCN import graph_sum_pool


class CustomGCN(D4CMPP2.MolecularNetwork):
    model_name = "custom_gcn"
    input_contract = D4CMPP2.InputContract(
        required=("compound_graphs", "compound_node_feature")
    )
    hyperparameters = {
        "hidden_dim": D4CMPP2.Hyperparameter(
            "int",
            default=64,
            low=1,
            search_low=16,
            search_high=256,
            step=16,
            grid=(32, 64, 128),
            description="Width of the graph representation.",
        ),
        "dropout": D4CMPP2.Hyperparameter(
            "float",
            default=0.1,
            low=0.0,
            high=0.5,
            search_low=0.0,
            search_high=0.5,
            grid=(0.0, 0.1, 0.2, 0.3),
            description="Dropout probability before prediction.",
        ),
        "huber_delta": D4CMPP2.Hyperparameter(
            "float",
            default=1.0,
            low=0.01,
            search_low=0.1,
            search_high=2.0,
            grid=(0.5, 1.0, 1.5),
            description="Transition point of the model-specific Huber loss.",
        ),
    }
    default_optimization_space = ("hidden_dim", "dropout", "huber_delta")

    def __init__(self, config):
        super().__init__(config)
        self.node_projection = nn.Linear(
            self.config["node_dim"], self.config["hidden_dim"]
        )
        self.dropout = nn.Dropout(self.config["dropout"])
        self.output = nn.Linear(
            self.config["hidden_dim"], self.config["target_dim"]
        )

    def forward(self, **batch):
        self.validate_input(batch)
        node = torch.relu(
            self.node_projection(batch["compound_node_feature"])
        )
        graph = graph_sum_pool(batch["compound_graphs"], node)
        return self.output(self.dropout(graph))

    def compute_loss(self, prediction, target):
        mask = ~torch.isnan(target)
        if not torch.any(mask):
            raise ValueError("A batch contains no finite target values for loss.")
        return nn.functional.huber_loss(
            prediction[mask],
            target[mask],
            delta=self.config["huber_delta"],
        )


def register():
    """Register once before calling D4CMPP2.train or D4CMPP2.optimize."""
    return D4CMPP2.register_network(CustomGCN, data_contract="molecule")


if __name__ == "__main__":
    register()
    print("Registered custom_gcn")
