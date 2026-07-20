import torch
import torch.nn as nn

from D4CMPP2.networks.base import (
    Hyperparameter,
    InputContract,
    ISA_HYPERPARAMETERS,
    ISA_OPTIMIZATION_SPACE,
    MolecularNetwork,
)
from D4CMPP2.networks.src.ISAT import ISATconvolution_PM
from D4CMPP2.networks.src.Linear import Linears
from D4CMPP2.networks.src.BiDropout import Bi_Dropout
from D4CMPP2.networks.src.GCN import graph_sum_pool
from D4CMPP2.networks.src.pyg_hetero import relation_graph

class ISATPN(MolecularNetwork):
    """Positive-negative ISAT with contribution-distribution regularization."""

    model_name = "isatpn"
    required_config = ("node_dim", "edge_dim", "target_dim")
    input_contract = InputContract(
        required=(
            "compound_graphs",
            "compound_r_node",
            "compound_i_node",
            "compound_r2r_edge",
            "compound_d2d_edge",
        ),
        optional=("get_feature", "get_score"),
    )
    hyperparameters = {
        **ISA_HYPERPARAMETERS,
        "score_variance_target": Hyperparameter(
            "float", default=0.17, low=0.0, search_low=0.01,
            search_high=1.0, grid=(0.1, 0.17, 0.25),
            description="Target variance of positive and negative scores.",
        ),
        "score_mean_target": Hyperparameter(
            "float", default=0.5, low=0.0, high=1.0,
            search_low=0.0, search_high=1.0, grid=(0.3, 0.5, 0.7),
            description="Target mean of positive and negative scores.",
        ),
        "score_magnitude_weight": Hyperparameter(
            "float", default=0.09, low=0.0, search_low=0.0,
            search_high=0.2, grid=(0.0, 0.03, 0.09, 0.15),
            description="Weight of score magnitude regularization.",
        ),
        "score_variance_weight": Hyperparameter(
            "float", default=0.03, low=0.0, search_low=0.0,
            search_high=0.2, grid=(0.0, 0.03, 0.1),
            description="Weight of score variance regularization.",
        ),
        "score_mean_weight": Hyperparameter(
            "float", default=0.03, low=0.0, search_low=0.0,
            search_high=0.2, grid=(0.0, 0.03, 0.1),
            description="Weight of score mean regularization.",
        ),
        "bidirectional_dropout": Hyperparameter(
            "float", default=0.4, low=0.0, high=0.5,
            search_low=0.0, search_high=0.5,
            grid=(0.0, 0.2, 0.4),
            description="Probability of suppressing either contribution branch.",
        ),
    }
    default_optimization_space = (
        *ISA_OPTIMIZATION_SPACE,
        "score_magnitude_weight",
        "score_variance_weight",
        "score_mean_weight",
        "bidirectional_dropout",
    )

    def __init__(self, config):
        aliases = {
            "score_variance_target": "score_var",
            "score_mean_target": "score_mean",
            "score_magnitude_weight": "alpha",
            "score_variance_weight": "beta",
            "score_mean_weight": "gamma",
        }
        normalized = dict(config)
        for new_name, old_name in aliases.items():
            if new_name not in normalized and old_name in normalized:
                normalized[new_name] = normalized[old_name]
        super().__init__(normalized)
        if self.config["target_dim"] != 1:
            raise ValueError(
                "ISATPN requires target_dim=1 because positive and negative "
                "branches are subtracted into one prediction."
            )
        
        
        hidden_dim = self.config["hidden_dim"]
        gcn_layers = self.config["conv_layers"]
        dropout = self.config["dropout"]
        linear_layers = self.config["linear_layers"]
        last_linear_dim = int(hidden_dim/(2**linear_layers))
        if last_linear_dim < 1:
            raise ValueError(
                "ISATPN requires hidden_dim >= 2 ** linear_layers; got "
                f"hidden_dim={hidden_dim}, linear_layers={linear_layers}."
            )

        self.embedding_rnode_lin = nn.Sequential(
            nn.Linear(self.config["node_dim"], hidden_dim, bias=False)
        )
        self.embedding_inode_lin = nn.Sequential(
            nn.Linear(1, hidden_dim, bias=False)
        )
        self.embedding_edge_lin = nn.Sequential(
            nn.Linear(self.config["edge_dim"], hidden_dim, bias=False)
        )
        self.ISATconv_PM = ISATconvolution_PM(hidden_dim, hidden_dim, hidden_dim, nn.LeakyReLU(), gcn_layers, dropout, False, True, 0.1)

        self.linears = Linears(hidden_dim,last_linear_dim, nn.ReLU(), linear_layers, dropout, False, False)
        self.out_linear = nn.Linear(last_linear_dim, self.config["target_dim"])

        self.Bi_Dropout=Bi_Dropout(self.config["bidirectional_dropout"])

        self.score_var = self.config["score_variance_target"]
        self.score_mean = self.config["score_mean_target"]
        self.alpha = self.config["score_magnitude_weight"]
        self.beta = self.config["score_variance_weight"]
        self.gamma = self.config["score_mean_weight"]
        
    
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
            raise ValueError(f"ISATPN input is missing required fields {missing!r}.")

        self.Bi_Dropout.set_drop_p(self.training)
        
        r_node = r_node.float()
        r_node = self.embedding_rnode_lin(r_node)
        i_node = i_node.float()
        i_node = self.embedding_inode_lin(i_node)
        r_edge = r_edge.float()
        r_edge = self.embedding_edge_lin(r_edge)
        
        real_graph = relation_graph(graph, 'r_nd', 'r2r')

        r_node_P, r_node_M, score_P, score_M = self.ISATconv_PM(graph, r_node, r_edge, i_node, d_edge, **kargs)
        if kargs.get('get_feature',False):
            return {'feature_P':r_node_P,'feature_N':r_node_M,'positive':score_P,'negative':score_M}

        h_P = graph_sum_pool(real_graph, r_node_P)
        h_P = self.linears(h_P)
        h_P = self.out_linear(h_P)

        h_M = graph_sum_pool(real_graph, r_node_M)
        h_M = self.linears(h_M)
        h_M = self.out_linear(h_M)

        h=torch.concat([h_P,h_M],axis=1)
        h=self.Bi_Dropout(h)
        output=torch.unsqueeze(h[:,0]-h[:,1],axis=1)
        if kargs.get('get_score',False):
            return {'prediction':output, 'positive':score_P,'negative':score_M}
        
        self.p_score_var = self.ISATconv_PM.p_score_var
        self.n_score_var = self.ISATconv_PM.n_score_var
        self.p_score_mean = self.ISATconv_PM.p_score_mean
        self.n_score_mean = self.ISATconv_PM.n_score_mean
        self.p_score_ms = self.ISATconv_PM.p_score_ms
        self.n_score_ms = self.ISATconv_PM.n_score_ms

        return output
        
    def compute_loss(self, scores, targets):
        mask = ~torch.isnan(targets)
        if not torch.any(mask):
            raise ValueError("A batch contains no finite target values for loss.")
        scores,targets=self.Bi_Dropout.drop_label(scores[mask],targets[mask])
        return torch.mean(torch.square(scores-targets))+\
                (self.p_score_ms+self.n_score_ms)*self.alpha +\
                (torch.abs(self.score_var-self.p_score_var)+torch.abs(self.score_var-self.n_score_var))*self.beta +\
                (torch.abs(self.score_mean-self.p_score_mean)+torch.abs(self.score_mean-self.n_score_mean))*self.gamma


network = ISATPN


