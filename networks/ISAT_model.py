import torch.nn as nn

from D4CMPP2.networks.base import (
    InputContract,
    ISA_HYPERPARAMETERS,
    ISA_OPTIMIZATION_SPACE,
    MolecularNetwork,
)
from D4CMPP2.networks.src.ISAT import ISATconvolution
from D4CMPP2.networks.src.Linear import Linears
from D4CMPP2.networks.src.GCN import graph_sum_pool
from D4CMPP2.networks.src.pyg_hetero import relation_graph

class ISAT(MolecularNetwork):
    """Interpretable structure-attention network."""

    model_name = "isat"
    required_config = ("node_dim", "edge_dim", "target_dim")
    input_contract = InputContract(
        required=(
            "compound_graphs",
            "compound_r_node",
            "compound_i_node",
            "compound_r2r_edge",
            "compound_d2d_edge",
        ),
        optional=("get_score",),
    )
    hyperparameters = ISA_HYPERPARAMETERS
    default_optimization_space = ISA_OPTIMIZATION_SPACE

    def __init__(self, config):
        super().__init__(config)

        hidden_dim = self.config["hidden_dim"]
        gcn_layers = self.config["conv_layers"]
        dropout = self.config["dropout"]
        linear_layers = self.config["linear_layers"]
        target_dim = self.config["target_dim"]

        self.embedding_rnode_lin = nn.Sequential(
            nn.Linear(self.config["node_dim"], hidden_dim, bias=False)
        )
        self.embedding_inode_lin = nn.Sequential(
            nn.Linear(1, hidden_dim, bias=False)
        )
        self.embedding_edge_lin = nn.Sequential(
            nn.Linear(self.config["edge_dim"], hidden_dim, bias=False)
        )
        self.ISATconv = ISATconvolution(hidden_dim, hidden_dim, hidden_dim, nn.LeakyReLU(), gcn_layers,dropout, False, True, 0.1)
        
        self.linears = Linears(hidden_dim, target_dim, nn.ReLU(), linear_layers, dropout, False, False, last=True)
    
    def forward(self, **kargs):
        graph = kargs.get('compound_graphs', kargs.get('graph'))
        r_node = kargs.get('compound_r_node', kargs.get('r_node'))
        i_node = kargs.get('compound_i_node', kargs.get('i_node'))
        r_edge = kargs.get('compound_r2r_edge', kargs.get('r_edge'))
        d_edge = kargs.get('compound_d2d_edge', kargs.get('d_edge'))
        missing = [
            name
            for name, value in {
                "compound_graphs": graph,
                "compound_r_node": r_node,
                "compound_i_node": i_node,
                "compound_r2r_edge": r_edge,
                "compound_d2d_edge": d_edge,
            }.items()
            if value is None
        ]
        if missing:
            raise ValueError(f"ISAT input is missing required fields {missing!r}.")

        r_node = r_node.float()
        r_node = self.embedding_rnode_lin(r_node)
        i_node = i_node.float()
        i_node = self.embedding_inode_lin(i_node)
        r_edge = r_edge.float()
        r_edge = self.embedding_edge_lin(r_edge)
        
        r_node, score = self.ISATconv(graph, r_node, r_edge, i_node, d_edge)
        h = graph_sum_pool(relation_graph(graph, 'r_nd', 'r2r'), r_node)
        h = self.linears(h)

        if kargs.get('get_score',False):
            return {'prediction':h, 'positive':score}

        return h

network = ISAT
