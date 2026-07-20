import torch
import torch.nn as nn

from D4CMPP2.networks.base import (
    InputContract,
    MolecularNetwork,
    STANDARD_GRAPH_HYPERPARAMETERS,
    STANDARD_GRAPH_OPTIMIZATION_SPACE,
)
from D4CMPP2.networks.src.Linear import Linears
from D4CMPP2.networks.src.DMPNN import DMPNNLayer
from D4CMPP2.networks.src.GCN import graph_sum_pool


class DMPNN(MolecularNetwork):
    """Directed-edge message-passing network."""

    model_name = "dmpnn"
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
        conv_layers = self.config["conv_layers"]
        dropout = self.config["dropout"]
        linear_layers = self.config["linear_layers"]
        target_dim = self.config["target_dim"]

        self.embedding_node_lin = nn.Linear(self.config["node_dim"], hidden_dim, bias=True)
        self.embedding_edge_lin = nn.Linear(self.config["edge_dim"], hidden_dim, bias=True)
        self.init_h_func = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim, bias=True),
            nn.LeakyReLU()
        )
        self.W_a = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim, bias=True),
            nn.LeakyReLU()
        )
        self.dropout_layer = nn.Dropout(dropout)
        self.DMPNNLayer = nn.ModuleList([DMPNNLayer(hidden_dim,hidden_dim,hidden_dim,nn.LeakyReLU(),dropout) for _ in range(conv_layers)])
        self.Linears = Linears(hidden_dim, target_dim, nn.LeakyReLU(), linear_layers, dropout, last = True)
        
    def send_income_edge(self, edges):
        return {'mail': edges.data['feat']}

    def sum_income_edge(self, nodes):
        hidden_feats = self.W_a(torch.cat([nodes.data['feat'], torch.sum(nodes.mailbox['mail'], 1)], dim=-1))
        hidden_feats = self.dropout_layer(hidden_feats)
        return {'hidden_feats': hidden_feats}

    def forward(self, **kwargs):
        graph = kwargs.get('compound_graphs', kwargs.get('graph'))
        node_feats = kwargs.get('compound_node_feature', kwargs.get('node_feats'))
        edge_feats = kwargs.get('compound_edge_feature', kwargs.get('edge_feats'))
        missing = [
            name
            for name, value in {
                "compound_graphs": graph,
                "compound_node_feature": node_feats,
                "compound_edge_feature": edge_feats,
            }.items()
            if value is None
        ]
        if missing:
            raise ValueError(f"DMPNN input is missing required fields {missing!r}.")

        node = self.embedding_node_lin(node_feats)
        edge = self.embedding_edge_lin(edge_feats)

        direct_feats = None
        backward_feats = None
        for layer in self.DMPNNLayer:
            hidden_feats, direct_feats, backward_feats = layer(graph, node, edge, direct_feats, backward_feats)

        graph_feats = graph_sum_pool(graph, hidden_feats)
        output = self.Linears(graph_feats)
        return output

network = DMPNN


