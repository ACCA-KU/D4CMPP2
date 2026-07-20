import torch.nn as nn

from D4CMPP2.networks.base import (
    Hyperparameter,
    InputContract,
    MolecularNetwork,
)
from D4CMPP2.networks.src.GCN import GCNs, graph_sum_pool
from D4CMPP2.networks.src.Linear import Linears


class GCN(MolecularNetwork):
    """Graph convolution using transformed self-plus-neighbor sums."""

    model_name = "gcn"
    input_contract = InputContract(
        required=("compound_graphs", "compound_node_feature"),
    )
    hyperparameters = {
        "hidden_dim": Hyperparameter(
            "int",
            default=64,
            low=1,
            search_low=16,
            search_high=256,
            step=16,
            grid=(32, 64, 128, 256),
            description="Hidden node and graph representation width.",
        ),
        "conv_layers": Hyperparameter(
            "int",
            default=4,
            low=1,
            search_low=1,
            search_high=8,
            step=1,
            grid=(2, 4, 6, 8),
            description="Number of graph convolution layers.",
        ),
        "linear_layers": Hyperparameter(
            "int",
            default=2,
            low=1,
            search_low=1,
            search_high=5,
            step=1,
            grid=(1, 2, 3, 4),
            description="Number of prediction-head linear layers.",
        ),
        "dropout": Hyperparameter(
            "float",
            default=0.2,
            low=0.0,
            high=0.5,
            search_low=0.0,
            search_high=0.5,
            grid=(0.0, 0.1, 0.2, 0.3, 0.5),
            description="Dropout probability.",
        ),
    }
    default_optimization_space = (
        "hidden_dim",
        "conv_layers",
        "linear_layers",
        "dropout",
    )

    def __init__(self, config):
        super().__init__(config)
        hidden_dim = self.config["hidden_dim"]
        self.node_embedding = nn.Linear(self.config["node_dim"], hidden_dim)
        # Keep the established module registration names so saved state dictionaries
        # remain readable while the public class/config contract is redesigned.
        self.GCNs = GCNs(
            hidden_dim,
            hidden_dim,
            hidden_dim,
            nn.ReLU(),
            self.config["conv_layers"],
            self.config["dropout"],
            False,
            True,
        )
        self.Linears = Linears(
            hidden_dim,
            self.config["target_dim"],
            nn.ReLU(),
            self.config["linear_layers"],
            self.config["dropout"],
            False,
            False,
            True,
        )

    @property
    def convolutions(self):
        return self.GCNs

    @property
    def prediction_head(self):
        return self.Linears

    def forward(self, **batch):
        self.validate_input(batch)
        graph = batch["compound_graphs"]
        node = self.node_embedding(batch["compound_node_feature"])
        node = self.GCNs(graph, node)
        graph_feature = graph_sum_pool(graph, node)
        return self.Linears(graph_feature)


# Staged bridge for code paths not yet migrated to the typed registry.
network = GCN
