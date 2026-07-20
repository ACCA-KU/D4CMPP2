import torch.nn as nn
from D4CMPP2.networks.base import (
    InputContract,
    MolecularNetwork,
    STANDARD_GRAPH_HYPERPARAMETERS,
    STANDARD_GRAPH_OPTIMIZATION_SPACE,
)
from D4CMPP2.networks.src.MPNN import MPNNs
from D4CMPP2.networks.src.GCN import graph_sum_pool
from D4CMPP2.networks.src.Linear import Linears

class MPNN(MolecularNetwork):
    """Edge-aware message-passing network with graph-sum readout."""

    model_name = "mpnn"
    required_config = ("node_dim", "edge_dim", "target_dim")
    input_contract = InputContract(
        required=(
            "compound_graphs",
            "compound_node_feature",
            "compound_edge_feature",
        )
    )
    hyperparameters = STANDARD_GRAPH_HYPERPARAMETERS
    default_optimization_space = STANDARD_GRAPH_OPTIMIZATION_SPACE

    def __init__(self, config):
        super().__init__(config)
        
        hidden_dim = self.config["hidden_dim"]
        mpnn_layers = self.config["conv_layers"]
        dropout = self.config["dropout"]
        linear_layers = self.config["linear_layers"]
        target_dim = self.config["target_dim"]


        self.node_embedding = nn.Linear(self.config["node_dim"], hidden_dim)
        self.edge_embedding = nn.Linear(self.config["edge_dim"], hidden_dim)

        self.MPNNs = MPNNs(hidden_dim, hidden_dim, hidden_dim, hidden_dim, nn.LeakyReLU(), mpnn_layers, dropout, False, True) # in_feats, hidden_feats, out_feats, activation, n_layers, dropout=0.2, batch_norm=False, residual_sum = False):
        self.Linears = Linears(hidden_dim,target_dim, nn.LeakyReLU(), linear_layers, dropout, False, False, last=True) # in_feats, out_feats, activation, n_layers, dropout=0.2, batch_norm=False, residual_sum = False):

    def forward(self, **kwargs):
        self.validate_input(kwargs)
        graph = kwargs["compound_graphs"]
        node_feats = kwargs["compound_node_feature"]
        edge_feats = kwargs["compound_edge_feature"]

        h = self.node_embedding(node_feats)
        e = self.edge_embedding(edge_feats)
        h = self.MPNNs(graph, h, e)
        h = graph_sum_pool(graph, h)
        h = self.Linears(h)
        return h

network = MPNN
