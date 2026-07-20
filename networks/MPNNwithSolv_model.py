import torch.nn as nn
from D4CMPP2.networks.base import (
    InputContract,
    MolecularNetwork,
    STANDARD_SOLVENT_HYPERPARAMETERS,
    STANDARD_SOLVENT_OPTIMIZATION_SPACE,
)
from D4CMPP2.networks.src.MPNN import MPNNs
from D4CMPP2.networks.src.GCN import graph_sum_pool
from D4CMPP2.networks.src.Linear import Linears
from D4CMPP2.networks.src.SolventLayer import SolventLayer

class SolventMPNN(MolecularNetwork):
    """Edge-aware compound MPNN coupled to a solvent graph branch."""

    model_name = "mpnn_solvent"
    required_config = ("node_dim", "edge_dim", "target_dim")
    input_contract = InputContract(
        required=(
            "compound_graphs",
            "compound_node_feature",
            "compound_edge_feature",
            "solvent_graphs",
            "solvent_node_feature",
        )
    )
    hyperparameters = STANDARD_SOLVENT_HYPERPARAMETERS
    default_optimization_space = STANDARD_SOLVENT_OPTIMIZATION_SPACE

    def __init__(self, config):
        super().__init__(config)
        
        hidden_dim = self.config["hidden_dim"]
        mpnn_layers = self.config["conv_layers"]
        dropout = self.config["dropout"]
        linear_layers = self.config["linear_layers"]

        self.node_embedding = nn.Linear(self.config["node_dim"], hidden_dim)
        self.edge_embedding = nn.Linear(self.config["edge_dim"], hidden_dim)
        self.MPNNs = MPNNs(hidden_dim, hidden_dim, hidden_dim, hidden_dim, nn.ReLU(), mpnn_layers, dropout, False, True) # in_feats, hidden_feats, out_feats, activation, n_layers, dropout=0.2, batch_norm=False, residual_sum = False):
        self.Linears = Linears(hidden_dim,hidden_dim, nn.ReLU(), linear_layers, dropout, False, False) # in_feats, out_feats, activation, n_layers, dropout=0.2, batch_norm=False, residual_sum = False):
        self.solvlayer = SolventLayer(self.config)


    def forward(self, **kwargs):
        graph = kwargs.get('compound_graphs', kwargs.get('graph'))
        node_feats = kwargs.get('compound_node_feature', kwargs.get('node_feats'))
        edge_feats = kwargs.get('compound_edge_feature', kwargs.get('edge_feats'))
        solv_graph = kwargs.get('solvent_graphs', kwargs.get('solv_graph'))
        solv_node_feats = kwargs.get('solvent_node_feature', kwargs.get('solv_node_feats'))
        values = {
            "compound_graphs": graph,
            "compound_node_feature": node_feats,
            "compound_edge_feature": edge_feats,
            "solvent_graphs": solv_graph,
            "solvent_node_feature": solv_node_feats,
        }
        missing = [name for name, value in values.items() if value is None]
        if missing:
            raise ValueError(
                f"SolventMPNN input is missing required fields {missing!r}."
            )

        h = self.node_embedding(node_feats)
        e = self.edge_embedding(edge_feats)
        h = self.MPNNs(graph, h, e)
        h = graph_sum_pool(graph, h)
        h = self.Linears(h)
        h = self.solvlayer(h, solv_graph, solv_node_feats)
        return h

network = SolventMPNN
