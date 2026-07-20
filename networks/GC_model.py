import torch.nn as nn

from D4CMPP2.networks.base import (
    Hyperparameter,
    InputContract,
    MolecularNetwork,
    STANDARD_GRAPH_HYPERPARAMETERS,
    STANDARD_GRAPH_OPTIMIZATION_SPACE,
)
from D4CMPP2.networks.src.GC import GCconvolution
from D4CMPP2.networks.src.GCN import graph_sum_pool
from D4CMPP2.networks.src.pyg_hetero import relation_graph

class GroupContributionNetwork(MolecularNetwork):
    """Group-contribution network over the ISA heterogeneous graph."""

    model_name = "group_contribution"
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
    hyperparameters = {
        **{
            name: field
            for name, field in STANDARD_GRAPH_HYPERPARAMETERS.items()
            if name != "linear_layers"
        },
        "activation": Hyperparameter(
            "categorical",
            default="Tanh",
            values=("Tanh", "Sigmoid", "ReLU", "Identity"),
            grid=("Tanh", "Sigmoid", "ReLU"),
            description="Activation applied to the normalized contribution sum.",
        ),
    }
    default_optimization_space = (
        *(
            name
            for name in STANDARD_GRAPH_OPTIMIZATION_SPACE
            if name != "linear_layers"
        ),
        "activation",
    )

    def __init__(self, config):
        if "activation" not in config and "func_f" in config:
            config = {**config, "activation": config["func_f"]}
        super().__init__(config)
        if self.config["target_dim"] != 1:
            raise ValueError(
                "GroupContributionNetwork requires target_dim=1 because its "
                "contribution head produces one scalar per molecule."
            )

        hidden_dim = self.config["hidden_dim"]
        gcn_layers = self.config["conv_layers"]
        dropout = self.config["dropout"]

        self.embedding_rnode_lin = nn.Sequential(
            nn.Linear(self.config["node_dim"], hidden_dim, bias=False)
        )
        self.embedding_inode_lin = nn.Sequential(
            nn.Linear(1, hidden_dim, bias=False)
        )
        self.embedding_edge_lin = nn.Sequential(
            nn.Linear(self.config["edge_dim"], hidden_dim, bias=False)
        )

        self.GCconv = GCconvolution(hidden_dim, hidden_dim, hidden_dim, nn.LeakyReLU(), gcn_layers, dropout, False, True, 0.1)
        
        self.linear = nn.Linear(1,1)
        self.normalize = nn.LayerNorm(1)
        self.func_f = getattr(nn, self.config["activation"])()
    
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
            raise ValueError(
                f"GroupContributionNetwork input is missing required fields {missing!r}."
            )

        r_node = r_node.float()
        r_node = self.embedding_rnode_lin(r_node)
        i_node = i_node.float()
        i_node = self.embedding_inode_lin(i_node)
        r_edge = r_edge.float()
        r_edge = self.embedding_edge_lin(r_edge)
        
        score = self.GCconv(graph, r_node, r_edge, i_node, d_edge)
        h = graph_sum_pool(relation_graph(graph, 'd_nd', 'd2d'), score)
        h = self.normalize(h)
        h = self.func_f(h)
        h = self.linear(h)
        
        if kargs.get('get_score',False):
            return {'prediction':h, 'positive':score}

        return h

network = GroupContributionNetwork
