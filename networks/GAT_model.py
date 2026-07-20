import torch.nn as nn
from D4CMPP2.networks.base import (
    InputContract,
    MolecularNetwork,
    STANDARD_GRAPH_HYPERPARAMETERS,
    STANDARD_GRAPH_OPTIMIZATION_SPACE,
)
from D4CMPP2.networks.src.GAT import GATs
from D4CMPP2.networks.src.GCN import graph_sum_pool
from D4CMPP2.networks.src.Linear import Linears

class GAT(MolecularNetwork):
    """Graph-attention network with graph-sum readout."""

    model_name = "gat"
    input_contract = InputContract(
        required=("compound_graphs", "compound_node_feature")
    )
    hyperparameters = STANDARD_GRAPH_HYPERPARAMETERS
    default_optimization_space = STANDARD_GRAPH_OPTIMIZATION_SPACE

    def __init__(self, config):
        super().__init__(config)
        
        hidden_dim = self.config["hidden_dim"]
        gcn_layers = self.config["conv_layers"]
        dropout = self.config["dropout"]
        linear_layers = self.config["linear_layers"]
        target_dim = self.config["target_dim"]


        self.node_embedding = nn.Linear(self.config["node_dim"], hidden_dim)

        self.GATs = GATs(hidden_dim, hidden_dim, hidden_dim, nn.ReLU(), gcn_layers, dropout, False, True) # in_feats, hidden_feats, out_feats, activation, n_layers, dropout=0.2, batch_norm=False, residual_sum = False):
        self.Linears = Linears(hidden_dim,target_dim, nn.ReLU(), linear_layers, dropout, False, False, True) # in_feats, out_feats, activation, n_layers, dropout=0.2, batch_norm=False, residual_sum = False):

    def forward(self, **kwargs):
        self.validate_input(kwargs)
        graph = kwargs["compound_graphs"]
        node_feats = kwargs["compound_node_feature"]

        h = self.node_embedding(node_feats)
        h = self.GATs(graph, h)
        h = graph_sum_pool(graph, h)
        h = self.Linears(h)
        return h

network = GAT
