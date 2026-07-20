import torch.nn as nn

from D4CMPP2.networks.base import (
    InputContract,
    MolecularNetwork,
    STANDARD_GRAPH_HYPERPARAMETERS,
    STANDARD_GRAPH_OPTIMIZATION_SPACE,
)
from D4CMPP2.networks.src.AFP import  AttentiveFP
from D4CMPP2.networks.src.Linear import Linears


class AFP(MolecularNetwork):
    """Attentive fingerprint network with optional atom-attention output."""

    model_name = "afp"
    required_config = ("node_dim", "edge_dim", "target_dim")
    input_contract = InputContract(
        required=(
            "compound_graphs",
            "compound_node_feature",
            "compound_edge_feature",
        ),
        optional=("get_score",),
    )
    hyperparameters = STANDARD_GRAPH_HYPERPARAMETERS
    default_optimization_space = STANDARD_GRAPH_OPTIMIZATION_SPACE

    def __init__(self, config):
        super().__init__(config)

        linear_layers = self.config["linear_layers"]
        dropout = self.config["dropout"]
        hidden_dim = self.config["hidden_dim"]

        self.embedding_node_lin = nn.Linear(self.config["node_dim"], hidden_dim, bias=True)
        self.embedding_edge_lin = nn.Linear(self.config["edge_dim"], hidden_dim, bias=True)

        self.AttentiveFP = AttentiveFP(self.config)
        self.Linears = Linears(hidden_dim,self.config["target_dim"], nn.ReLU(), linear_layers, dropout, False, False, True) 
        

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
            raise ValueError(f"AFP input is missing required fields {missing!r}.")

        node = self.embedding_node_lin(node_feats)
        edge = self.embedding_edge_lin(edge_feats)

        super_node, att_w = self.AttentiveFP(graph, node, edge)

        output = self.Linears(super_node)
        
        if kwargs.get('get_score',False):
            return {'prediction':output, 'positive':att_w}
        
        return output


network = AFP
